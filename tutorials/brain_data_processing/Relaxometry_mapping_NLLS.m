clear; clc;

%% Inputs
load I4D_NESMA_cropped.mat;
load I4D_raw_cropped.mat;
[dim1,dim2,dim3,dim4]=size(I4D_NESMA_cropped);
TE=11.32:11.32:11.32*32;

%% Values of k to process
k_values = [3,5,8];

%% Optimization options
options = optimset('Display','off');

for kidx = 1:numel(k_values)
    tic;
    k = k_values(kidx);
    fprintf('Processing NLLS ... Slice # %d (k = %d of %d)\n', k, kidx, numel(k_values));

    %% Preallocate 2D maps for this slice (we only need 2D maps for storing results of slice k)
    MWFs_NESMA_slice = single(zeros(dim1,dim2));
    MWFl_NESMA_slice = single(zeros(dim1,dim2));
    T2s_NESMA_slice = single(zeros(dim1,dim2));
    T2l_NESMA_slice = single(zeros(dim1,dim2));
    Offset_NESMA_slice = single(zeros(dim1,dim2));

    MWFs_raw_slice = single(zeros(dim1,dim2));
    MWFl_raw_slice = single(zeros(dim1,dim2));
    T2s_raw_slice = single(zeros(dim1,dim2));
    T2l_raw_slice = single(zeros(dim1,dim2));
    Offset_raw_slice = single(zeros(dim1,dim2));

    MWF_NESMA_mon_slice = single(zeros(dim1,dim2));
    T2s_NESMA_mon_slice = single(zeros(dim1,dim2));
    Offset_NESMA_mon_slice = single(zeros(dim1,dim2));

    MWF_raw_mon_slice = single(zeros(dim1,dim2));
    T2s_raw_mon_slice = single(zeros(dim1,dim2));
    Offset_raw_mon_slice = single(zeros(dim1,dim2));

    %% Mapping (process this single slice k)
    for i = 1:dim1
        for j = 1:dim2
            if I4D_NESMA_cropped(i,j,k,1) > 50
                % ---------- NESMA bi-exponential fit ----------
                y_NESMA = squeeze(I4D_NESMA_cropped(i,j,k,:));
                y_NESMA = y_NESMA / y_NESMA(1);
                init_offset = y_NESMA(1) - 0.2*exp(-TE(1)/20) - 0.8*exp(-TE(1)/80);
                P0 = [0.2 20 0.8 80 init_offset];
                Pi = lsqnonlin(@(P) fit_bi(P,y_NESMA,TE), P0, [0 0 0 0 0], [0.75 80 2 300 inf], options);
                MWFs_NESMA_slice(i,j) = Pi(1);
                T2s_NESMA_slice(i,j) = Pi(2);
                MWFl_NESMA_slice(i,j) = Pi(3);
                T2l_NESMA_slice(i,j) = Pi(4);
                Offset_NESMA_slice(i,j) = Pi(5);

                % ---------- RAW bi-exponential fit ----------
                y_raw = squeeze(I4D_raw_cropped(i,j,k,:));
                y_raw = y_raw / y_raw(1);
                Pi_raw = lsqnonlin(@(P) fit_bi(P,y_raw,TE), P0, [0 0 0 0 0], [0.75 80 2 300 inf], options);
                MWFs_raw_slice(i,j) = Pi_raw(1);
                T2s_raw_slice(i,j) = Pi_raw(2);
                MWFl_raw_slice(i,j) = Pi_raw(3);
                T2l_raw_slice(i,j) = Pi_raw(4);
                Offset_raw_slice(i,j) = Pi_raw(5);

                % ---------- NESMA mono-exponential fit ----------
                y_NESMA = squeeze(I4D_NESMA_cropped(i,j,k,:)); % reload to be safe
                y_NESMA = y_NESMA / y_NESMA(1);
                init_offset_mon = y_NESMA(1) - exp(-TE(1)/20);
                P0_mon = [1 20 init_offset_mon];
                Pi_mon = lsqnonlin(@(P) fit_mon(P,y_NESMA,TE), P0_mon, [0 0 0], [1.5 300 inf], options);
                MWF_NESMA_mon_slice(i,j) = Pi_mon(1);
                T2s_NESMA_mon_slice(i,j) = Pi_mon(2);
                Offset_NESMA_mon_slice(i,j) = Pi_mon(3);

                % ---------- RAW mono-exponential fit ----------
                y_raw = squeeze(I4D_raw_cropped(i,j,k,:));
                y_raw = y_raw / y_raw(1);
                Pi_mon_raw = lsqnonlin(@(P) fit_mon(P,y_raw,TE), P0_mon, [0 0 0], [1.5 300 inf], options);
                MWF_raw_mon_slice(i,j) = Pi_mon_raw(1);
                T2s_raw_mon_slice(i,j) = Pi_mon_raw(2);
                Offset_raw_mon_slice(i,j) = Pi_mon_raw(3);
            end
        end
    end
    toc;
    %% skull stripping
    % Per your instruction load the skull-strip file per k:
    skull_fname = sprintf('raw_cropped_slice%d.mat', k);
    if exist(skull_fname,'file')
        S = load(skull_fname);
        fn = fieldnames(S);
        % assume the image is the first field, else adapt accordingly
        slice_oi = S.(fn{1});
        % user said "thresholding each of these under 700 gives you the skull stripping filter."
        % so skull filter is true where value < 700
        skull_strip_filter = slice_oi(:,:,1) > 700;
    else
        warning('Skull strip file %s not found. Creating empty (false) skull_strip_filter.', skull_fname);
        skull_strip_filter = false(dim1,dim2);
    end

    %% bic/aic calculations for NESMA (for this slice)
    bic_vals_mon = single(zeros(dim1,dim2));
    bic_vals_bi  = single(zeros(dim1,dim2));
    aic_vals_mon = single(zeros(dim1,dim2));
    aic_vals_bi  = single(zeros(dim1,dim2));
    residuals_mon = single(zeros(dim1,dim2));
    residuals_bi  = single(zeros(dim1,dim2));
    tic;
    for i=1:dim1
        for j=1:dim2
            y_NESMA = squeeze(I4D_NESMA_cropped(i,j,k,:));
            y_NESMA = y_NESMA / y_NESMA(1);
            bi_params = [MWFs_NESMA_slice(i,j), T2s_NESMA_slice(i,j), ...
                         MWFl_NESMA_slice(i,j), T2l_NESMA_slice(i,j), ...
                         Offset_NESMA_slice(i,j)];
            mon_params = [MWF_NESMA_mon_slice(i,j), T2s_NESMA_mon_slice(i,j), ...
                          Offset_NESMA_mon_slice(i,j)];
            bi_residual = norm(fit_bi(bi_params, y_NESMA, TE))^2;
            mon_residual = norm(fit_mon(mon_params, y_NESMA, TE))^2;
            residuals_mon(i,j) = mon_residual;
            residuals_bi(i,j)  = bi_residual;
            n = length(y_NESMA);
            % BIC/AIC as in original script
            bic_vals_mon(i,j) = n*log(mon_residual/(n-3)) + 3*log(n);
            bic_vals_bi(i,j)  = n*log(bi_residual/(n-5)) + 5*log(n);
            aic_vals_mon(i,j) = 2*3 + n*log(mon_residual/n);
            aic_vals_bi(i,j)  = 2*5 + n*log(bi_residual/n);
        end
    end
    toc;
    bic_mask_with_skull = bic_vals_mon - bic_vals_bi < 0.0;
    bic_mask = double(bic_mask_with_skull);
    bic_mask(~skull_strip_filter) = 0.0;

    aic_mask_with_skull = aic_vals_mon - aic_vals_bi < 0.0;
    aic_mask = double(aic_mask_with_skull);
    aic_mask(~skull_strip_filter) = 0.0;

    %% Prepare result struct and save - labelled by k
    NESMA_NLLS_results = struct();
    % NESMA mono results (2D maps for this slice)
    NESMA_NLLS_results.NESMA.mon.MWF   = MWF_NESMA_mon_slice;
    NESMA_NLLS_results.NESMA.mon.T2s   = T2s_NESMA_mon_slice;
    NESMA_NLLS_results.NESMA.mon.Offset= Offset_NESMA_mon_slice;

    % NESMA bi results
    NESMA_NLLS_results.NESMA.bi.MWFs  = MWFs_NESMA_slice;
    NESMA_NLLS_results.NESMA.bi.T2s   = T2s_NESMA_slice;
    NESMA_NLLS_results.NESMA.bi.MWFl  = MWFl_NESMA_slice;
    NESMA_NLLS_results.NESMA.bi.T2l   = T2l_NESMA_slice;
    NESMA_NLLS_results.NESMA.bi.Offset= Offset_NESMA_slice;

    % RAW results (mono + bi)
    NESMA_NLLS_results.RAW.mon.MWF    = MWF_raw_mon_slice;
    NESMA_NLLS_results.RAW.mon.T2s    = T2s_raw_mon_slice;
    NESMA_NLLS_results.RAW.mon.Offset = Offset_raw_mon_slice;

    NESMA_NLLS_results.RAW.bi.MWFs    = MWFs_raw_slice;
    NESMA_NLLS_results.RAW.bi.T2s     = T2s_raw_slice;
    NESMA_NLLS_results.RAW.bi.MWFl    = MWFl_raw_slice;
    NESMA_NLLS_results.RAW.bi.T2l     = T2l_raw_slice;
    NESMA_NLLS_results.RAW.bi.Offset  = Offset_raw_slice;

    % Add skull and masks
    NESMA_NLLS_results.skull_strip_filter = skull_strip_filter;
    NESMA_NLLS_results.bic_mask = bic_mask;
    NESMA_NLLS_results.aic_mask = aic_mask;

    outname = sprintf('NESMA_NLLS_results_slice%d.mat', k);
    save(outname, '-struct', 'NESMA_NLLS_results');
    fprintf('Saved results to %s\n', outname);

end % k loop

disp('All requested slices processed and saved.');

%% Fit functions (unchanged)
function G = fit_mon(P,Sexp,TE)
    Sth(1,:) = P(3) + P(1)*exp(-TE./P(2));
    G = Sexp - Sth';
end

function F = fit_bi(P,Sexp,TE)
    Sth(1,:) = P(5) + P(1)*exp(-TE./P(2)) + P(3)*exp(-TE./P(4));
    F = Sexp - Sth';
end

