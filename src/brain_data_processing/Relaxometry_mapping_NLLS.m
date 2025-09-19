clear;clc;

%% Inputs
load I4D_NESMA;
load I4D_raw;
[dim1,dim2,dim3,dim4]=size(I4D_NESMA);
TE=11.32:11.32:11.32*32;

%% Initialization
MWFs_NESMA=single(zeros(dim1,dim2,dim3));
MWFl_NESMA=single(zeros(dim1,dim2,dim3));
T2s_NESMA=single(zeros(dim1,dim2,dim3));
T2l_NESMA=single(zeros(dim1,dim2,dim3));
Offset_NESMA = single(zeros(dim1,dim2,dim3));


MWFs_raw=single(zeros(dim1,dim2,dim3));
MWFl_raw=single(zeros(dim1,dim2,dim3));
T2s_raw=single(zeros(dim1,dim2,dim3));
T2l_raw=single(zeros(dim1,dim2,dim3));
Offset_raw = single(zeros(dim1,dim2,dim3));

MWF_NESMA_mon=single(zeros(dim1,dim2,dim3));
T2s_NESMA_mon=single(zeros(dim1,dim2,dim3));
Offset_NESMA_mon = single(zeros(dim1,dim2,dim3));

MWF_raw_mon=single(zeros(dim1,dim2,dim3));
T2s_raw_mon=single(zeros(dim1,dim2,dim3));
Offset_raw_mon = single(zeros(dim1,dim2,dim3));




%% Mapping
options=optimset('Display','off');
for k=1:dim3
    disp(['NLLS ... Slice # ' num2str(k) ' of '  num2str(dim3)]);
    if k ~= 5
        continue
    end 
    for i=1:dim1
        tic
        for j=1:dim2
            if I4D_NESMA(i,j,k,1)>50
                y_NESMA(:,1)=I4D_NESMA(i,j,k,:);
                y_NESMA = y_NESMA/y_NESMA(1,1);
                init_offset = y_NESMA(1,1) - 0.2*exp(-TE(1)/20) - 0.8*exp(-TE(1)/80); 
                P0=[0.2 20 0.8 80 init_offset];
                Pi=lsqnonlin(@(P) fit_bi(P,y_NESMA,TE),P0,[0 0 0 0 0],[0.75 80 2 300 inf],options);
                MWFs_NESMA(i,j,k)=Pi(1);
                T2s_NESMA(i,j,k)=Pi(2);
                MWFl_NESMA(i,j,k) = Pi(3);
                T2l_NESMA(i,j,k)=Pi(4);
                Offset_NESMA(i,j,k) = Pi(5);

                y_raw(:,1)=I4D_raw(i,j,k,:);
                y_raw = y_raw/y_raw(1,1);
                Pi=lsqnonlin(@(P) fit_bi(P,y_raw,TE),P0,[0 0 0 0 0],[0.75 80 2 300 inf],options);
                MWFs_raw(i,j,k)=Pi(1);
                T2s_raw(i,j,k)=Pi(2);
                MWFl_raw(i,j,k) = Pi(3);
                T2l_raw(i,j,k)=Pi(4);
                Offset_raw(i,j,k) = Pi(5);

                y_NESMA(:,1)=I4D_NESMA(i,j,k,:);
                y_NESMA = y_NESMA/y_NESMA(1,1);
                init_offset_mon = y_NESMA(1,1) - exp(-TE(1)/20);
                P0_mon=[1 20 init_offset_mon];
                Pi_mon=lsqnonlin(@(P) fit_mon(P,y_NESMA,TE),P0_mon,[0 0 0],[1.5 300 inf],options);
                MWF_NESMA_mon(i,j,k)=Pi_mon(1);
                T2s_NESMA_mon(i,j,k)=Pi_mon(2);
                Offset_NESMA_mon(i,j,k) = Pi_mon(3);

                y_raw(:,1)=I4D_raw(i,j,k,:);
                y_NESMA = y_NESMA/y_NESMA(1,1);
                Pi_mon=lsqnonlin(@(P) fit_mon(P,y_raw,TE),P0_mon,[0 0 0],[1.5 300 inf],options);
                MWF_raw_mon(i,j,k)=Pi_mon(1);
                T2s_raw_mon(i,j,k)=Pi_mon(2);
                Offset_raw_mon(i,j,k) = Pi_mon(3);
            end
        end
        toc
    end
end

%% skull stripping  
k=5;
load rS_slice5.mat
skull_strip_filter = slice_oi(:,:,1) > 700;

%% bic/aic
bic_vals_mon = single(zeros(dim1,dim2));
bic_vals_bi = single(zeros(dim1,dim2));
aic_vals_mon = single(zeros(dim1,dim2));
aic_vals_bi = single(zeros(dim1,dim2));
residuals_mon = single(zeros(dim1,dim2)); 
residuals_bi = single(zeros(dim1,dim2));
for i=1:dim1
    for j=1:dim2
        y_NESMA(:,1)=I4D_NESMA(i,j,k,:);
        y_NESMA = y_NESMA/y_NESMA(1,1);
        bi_params = [MWFs_NESMA(i,j,k), T2s_NESMA(i,j,k), ...
                     MWFl_NESMA(i,j,k), T2l_NESMA(i,j,k), ...
                     Offset_NESMA(i,j,k)];
        mon_params = [MWF_NESMA_mon(i,j,k), T2s_NESMA_mon(i,j,k), ...
                      Offset_NESMA_mon(i,j,k)];
        bi_residual = norm(fit_bi(bi_params, y_NESMA, TE))^2;
        mon_residual = norm(fit_mon(mon_params, y_NESMA, TE))^2; 
        residuals_mon(i,j) = mon_residual;
        residuals_bi(i,j) = bi_residual; 
        n = size(y_NESMA,1); 
        bic_vals_mon(i,j) = n*log(mon_residual/(n-3)) + 3*log(n);
        bic_vals_bi(i,j) = n*log(bi_residual/(n-5)) + 5*log(n);
        aic_vals_mon(i,j) = 2*3 + n*log(mon_residual/n);
        aic_vals_bi(i,j) = 2*5 + n*log(bi_residual/n);
    end
end
bic_mask_with_skull = bic_vals_mon - bic_vals_bi < 0.0; 
bic_mask = bic_mask_with_skull + 0.0; 
bic_mask(~skull_strip_filter) = 0.0; 

aic_mask_with_skull = aic_vals_mon - aic_vals_bi < 0.0; 
aic_mask = aic_mask_with_skull + 0.0; 
aic_mask(~skull_strip_filter) = 0.0; 

%% save if needed
NESMA_NLLS_results = struct(); 
NESMA_NLLS_results.mon_results = mon_results;
NESMA_NLLS_results.bi_results = bi_results;
NESMA_NLLS_results.skull_strip_filter = skull_strip_filter;
NESMA_NLLS_results.bic_mask = bic_mask;
NESMA_NLLS_results.aic_mask = aic_mask; 
save('NESMA_NLLS_results.mat', '-struct', "NESMA_NLLS_results"); 

%%
s=5;
figure;
subplot(221);imagesc(MWF_NESMA_mon(:,:,s),[0 0.5]);colormap jet; axis off;colorbar;title('MWF (n.u.)');
subplot(222);imagesc(T2s_NESMA_mon(:,:,s),[0 140]);colormap jet; axis off;colorbar;title('T2s (ms)');
% subplot(223);imagesc(T2l_NESMA(:,:,s),[0 140]);colormap jet; axis off;colorbar;title('T2l (ms)');
subplot(224);imagesc(Amplitude_NESMA_mon(:,:,s), [0 1.2*1e3]);colormap jet; axis off;colorbar;title('Amplitudes');
sgtitle("NESMA Data - m41 - slice 5 (1-10)")

figure;
subplot(131);imagesc(MWF_raw(:,:,s),[0 0.4]);colormap jet; axis off;colorbar;title('MWF (n.u.)');
subplot(132);imagesc(T2s_raw(:,:,s),[0 60]);colormap jet; axis off;colorbar;title('T2s (ms)');
subplot(133);imagesc(T2l_raw(:,:,s),[0 140]);colormap jet; axis off;colorbar;title('T2l (ms)');
sgtitle("Raw Data - m41 - slice 5 (1-10)")

%% 

m41_data = struct();
m41_data.raw.T2l = T2l_raw;
m41_data.raw.T2s = T2s_raw;m41_data.raw.MWF = MWF_raw;

m41_data.NESMA.T2l = T2l_NESMA;
m41_data.NESMA.T2s = T2s_NESMA;
m41_data.NESMA.MWF = MWF_NESMA;

save('m41_dataStruct.mat', '-struct', 'm41_data');

%%

load("m41_dataStruct.mat")
T2l_NESMA = NESMA.T2l;
T2s_NESMA = NESMA.T2s;
MWF_NESMA = NESMA.MWF;

T2l_raw = raw.T2l;
T2s_raw = raw.T2s;
MWF_raw = raw.MWF;


%% fit mon
function G=fit_mon(P,Sexp,TE)

Sth(1,:)=P(3)+ P(1)*exp(-TE./P(2));
G=Sexp-Sth';

end

%% fit bi
function F=fit_bi(P,Sexp,TE)

Sth(1,:)=P(5)+P(1)*exp(-TE./P(2))+P(3)*exp(-TE./P(4));
F=Sexp-Sth';

end
