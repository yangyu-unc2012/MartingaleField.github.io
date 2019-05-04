# LeetCode

<!-- TOC depthFrom:2 -->

- [Array](#Array)
- [Linked List](#Linked-List)

<!-- /TOC -->

# Array

### Two Sum

Given an array of integers, return **indices** of the two numbers such that they add up to a specific target.
You may assume that each input would have **exactly one** solution, and you may not use the same element twice.

C++:
```c++
vector<int> twoSum(vector<int> &nums, int target) {
    vector<int> ans;
    if (nums.empty()) return ans;

    unordered_map<int, int> num_to_idx;
    for (int i = 0; i < nums.size(); ++i) {
        int gap = target - nums[i];
        if (num_to_idx.find(gap) != num_to_idx.end()) {
            ans.emplace_back(i);
            ans.emplace_back(num_to_idx[gap]);
            break;
        }
        num_to_idx.emplace(nums[i], i);
    }
    return ans;
}
```
Python3:
```python
def twoSum(nums: List[int], target: int) -> List[int]:
    num_to_idx = {}
    for i, num in enumerate(nums):
        gap = target - num
        if gap not in num_to_idx:
            num_to_idx[num] = i
        else:
            return i, num_to_idx[gap]
```

### Container With Most Water

Given `n` non-negative integers `a1, a2, ..., an` , where each represents a point at coordinate `(i, ai)`. `n` vertical lines are drawn such that the two endpoints of line `i` is at `(i, ai)` and `(i, 0)`. Find two lines, which together with x-axis forms a container, such that the container contains the most water.

*Solution*: 
Use two pointers. Pointer `i` points to the first element and `j` to the last. The water volume is `(j - i) * h` where `h = min(height[i], height[j])`.
* If there exists taller bar on the right of `i` than `h`, move `i` to it and check if we have a better result.
* If there exists taller bar on the left of `j` than `h`, move `j` to it and check if we have a better result.

C++
```c++
int maxArea(vector<int> &height) {
    int water = 0;
    int i = 0, j = height.size() - 1;
    while (i < j) {
        int h = min(height[i], height[j]);
        water = max(water, (j - i) * h);
        while (height[i] <= h && i < j) i++;
        while (height[j] <= h && i < j) j--;
    }
    return water;
}
```

Python3
```python
def maxArea(self, height: List[int]) -> int:
    i, j = 0, len(height) - 1
    ans = 0
    while i < j:
        h = min(height[i], height[j])
        ans = max(ans, (j - i) * h)
        while height[i] <= h and i < j:
            i += 1
        while height[j] <= h and i < j:
            j -= 1
    return ans
```

### 3Sum
Given an array nums of n integers, are there elements `a, b, c` in nums such that `a + b + c = 0?` Find all unique triplets in the array which gives the sum of zero.

**Note:**

The solution set must not contain duplicate triplets.

C++
```c++
vector<vector<int>> threeSum(vector<int> &nums) {
    vector<vector<int>> result;
    sort(nums.begin(), nums.end());
    int n = nums.size();
    for (int i = 0; i < n - 2; ++i) {
        if (i > 0 && nums[i] == nums[i - 1]) continue;
        int j = i + 1, k = n - 1;
        while (j < k) {
            int sum = nums[i] + nums[j] + nums[k];
            if (sum < 0) {
                ++j;
                while (j < k && nums[j] == nums[j - 1]) ++j;
            } else if (sum > 0) {
                --k;
                while (j < k && nums[k] == nums[k + 1]) --k;
            } else {
                result.push_back({nums[i], nums[j], nums[k]});
                ++j;
                --k;
                while (j < k && nums[j] == nums[j - 1] && nums[k] == nums[k + 1]) {
                    ++j;
                    --k;
                }
            }
        }
    }
    return result;
}
```

Python3
```python
def threeSum(nums: 'List[int]') -> 'List[List[int]]':
    ans = []
    nums.sort()
    n = len(nums)
    for i in range(n - 2):
        j, k = i + 1, n - 1
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        while j < k:
            sum = nums[i] + nums[j] + nums[k]
            if sum < 0:
                j += 1
                while j < k and nums[j] == nums[j - 1]:
                    j += 1
            elif sum > 0:
                k -= 1
                while j < k and nums[k] == nums[k + 1]:
                    k -= 1
            else:
                ans.append([nums[i], nums[j], nums[k]])
                j += 1
                k -= 1
                while j < k and nums[j] == nums[j - 1] and nums[k] == nums[k + 1]:
                    j += 1
                    k -= 1
    return ans
```

# Linked List