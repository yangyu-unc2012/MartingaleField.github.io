https://martingalefield.github.io/

# LeetCode

<!-- TOC depthFrom:2 -->

- [Array](#Array)
    - [Two Sum](#Two-Sum)
    - [Container With Most Water](#container-with-most-water)
    - [3Sum](#3Sum)
    - [3Sum Closest](#3Sum-closest)
    - [4Sum](#4sum)
    - [4Sum II](#4sum-ii)
    - [Remove Duplicates from Sorted Array](#Remove-Duplicates-from-Sorted-Array)
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
Given an array nums of n integers, are there elements `a, b, c` in nums such that `a + b + c = 0`? Find all unique triplets in the array which gives the sum of zero.

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
                result.push_back({nums[i], nums[j++], nums[k--]});
                while (j < k && nums[j] == nums[j - 1] && nums[k] == nums[k + 1])
                    ++j, --k;
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

### 3Sum Closest

Given an array nums of `n` integers and an integer target, find three integers in `nums` such that the sum is closest to `target`. Return the sum of the three integers. You may assume that each input would have exactly one solution.

C++
```c++
int threeSumClosest(vector<int> &nums, int target) {
    int res = nums[0] + nums[1] + nums[2], n = nums.size();
    sort(nums.begin(), nums.end());
    for (int i = 0; i < n; ++i) {
        int j = i + 1, k = n - 1;
        while (j < k) {
            int diff = target - nums[i] - nums[j] - nums[k];
            if (diff == 0)
                return target;
            if (abs(diff) < abs(res - target)) {
                res = nums[i] + nums[j] + nums[k];
            } else if (diff < 0) {
                k--;
            } else {
                j++;
            }
        }
    }
    return res;
}
```

Python3
```python
def threeSumClosest(nums: 'List[int]', target: 'int') -> 'int':
    nums.sort()
    ans = nums[0] + nums[1] + nums[2]
    n = len(nums)
    for i in range(n - 2):
        j, k = i + 1, n - 1
        while j < k:
            sum = nums[i] + nums[j] + nums[k]
            diff = target - sum
            if diff > 0:
                j += 1
            elif diff < 0:
                k -= 1
            else:
                ans = sum
                break
            if abs(diff) < abs(target - ans):
                ans = sum
    return ans
```

### 4Sum

Given an array `nums` of `n` integers and an integer `target`, are there elements `a`, `b`, `c`, and `d` in `nums` such that `a + b + c + d = target`? Find all unique quadruplets in the array which gives the sum of `target`.

**Note:**

The solution set must not contain duplicate quadruplets.

C++
```c++
vector<vector<int>> fourSum(vector<int> &nums, int target) {
    vector<vector<int>> result;
    int n = nums.size();
    if (n < 4) return result;
    sort(nums.begin(), nums.end());
    for (int a = 0; a < n - 3; ++a) {
        // Pruning
        if (nums[a] + nums[n - 1] + nums[n - 2] + nums[n - 3] < target ||
            nums[a] + nums[a + 1] + nums[a + 2] + nums[a + 3] > target ||
            (a > 0 && nums[a] == nums[a - 1]))
            continue;
        for (int b = a + 1; b < n - 2; ++b) {
            if (b > a + 1 && nums[b] == nums[b - 1])
                continue;
            int c = b + 1, d = n - 1;
            while (c < d) {
                int sum = nums[a] + nums[b] + nums[c] + nums[d];
                if (sum < target) {
                    c++;
                    while (c < d && nums[c] == nums[c - 1])
                        c++;
                } else if (sum > target) {
                    d--;
                    while (c < d && nums[d] == nums[d + 1])
                        d--;
                } else {
                    result.push_back({nums[a], nums[b], nums[c++], nums[d--]});
                    while (c < d && nums[c] == nums[c - 1] && nums[d] == nums[d + 1])
                        c++, d--;
                }
            }
        }
    }
    return result;
}
```

### 4Sum II

Given four lists `A, B, C, D` of integer values, compute how many tuples `(i, j, k, l)` there are such that `A[i] + B[j] + C[k] + D[l]` is zero.

To make problem a bit easier, all `A, B, C, D` have same length of `N` where `0 <= N <= 500`.

C++
```c++
int fourSumCount(vector<int> &A, vector<int> &B, vector<int> &C, vector<int> &D) {
    unordered_map<int, int> sum_freq;
    int ans = 0;
    for (int a : A)
        for (int b : B)
            sum_freq[a + b]++;
    for (int c : C)
        for (int d : D)
            if (sum_freq.count(-(c + d)))
                ans += sum_freq[-(c + d)];
    return ans;
}
```

### Remove Duplicates from Sorted Array

Given a sorted array `nums`, remove the duplicates in-place such that each element appear only once and return the new length.

Do not allocate extra space for another array, you must do this by **modifying the input array in-place** with `O(1)` extra memory.


C++
```c++
int removeDuplicates(vector<int> &nums) {
    if (nums.size() < 2)
        return nums.size();
    int j = 1;
    for (int i = 1; i < nums.size(); ++i) {
        if (nums[i] != nums[j - 1]) {
            nums[j++] = nums[i];
        }
    }
    return j;
}
```

### Remove Duplicates from Sorted Array II

Given a sorted array `nums`, remove the duplicates **in-place** such that duplicates appeared at most **twice** and return the new length.

Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.

*Solution*:
At first glance, we can follow the same idea as previous problem. Compare `nums[i]` with the current last two elements of the new array. If either of the comparison return false, we can update the new array. 

In fact, we simply need to compare `nums[i] == nums[j - 2]`. If this returns false, we can update the new array no matter what.
- If `nums[i] == nums[j - 1]`, since we allow at most two duplicates, we can copy `nums[i]` to the end of the new array.

C++
```c++
int removeDuplicates(vector<int> &nums) {
    if (nums.size() < 3)
        return nums.size();
    int j = 2;
    for (int i = 2; i < nums.size(); ++i) {
        if (nums[i] != nums[j - 2]) {
            nums[j++] = nums[i];
        }
    }
    return j;
}
```

# Linked List
