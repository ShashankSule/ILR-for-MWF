from typing import List
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        left = 0 
        curr = 0 
        ans = 0
        print(f'nums: {nums}')
        for right in range(len(nums)):
            if nums[right] == 0:
                curr = curr+1
                print(f'{curr} : curr')
            while curr > k and left < right: 
                if nums[left] == 0: 
                    curr = curr - 1
                left = left + 1 
                print(f'Updating left pointer...')
            ans = max(ans, right - left + 1)
        return ans
    
if __name__ == '__main__':
    solution = Solution()
    nums = [1,1,1,0,0,0,1,1,1,1,0]
    k = 2
    ans = solution.longestOnes(nums, k)
    print(ans)
