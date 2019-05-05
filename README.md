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
    - [Remove Duplicates from Sorted Array II](#Remove-Duplicates-from-Sorted-Array-ii)
    - [Find Missing Positive](#Find-missing-positive)
    - [Insert Interval](#insert-interval)
    - [Majority Element](#majority-element)
    - [Majority Element II](#majority-element-ii)
- [Linked List](#Linked-List)


<!-- /TOC -->

# Array


### Two Sum

Given an array of integers, return **indices** of the two numbers such that they add up to a specific target.
You may assume that each input would have **exactly one** solution, and you may not use the same element twice.

#### C++
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

#### Python3
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
---
### Container With Most Water

Given `n` non-negative integers `a1, a2, ..., an` , where each represents a point at coordinate `(i, ai)`. `n` vertical lines are drawn such that the two endpoints of line `i` is at `(i, ai)` and `(i, 0)`. Find two lines, which together with x-axis forms a container, such that the container contains the most water.

#### Solution 
Use two pointers. Pointer `i` points to the first element and `j` to the last. The water volume is `(j - i) * h` where `h = min(height[i], height[j])`.
* If there exists taller bar on the right of `i` than `h`, move `i` to it and check if we have a better result.
* If there exists taller bar on the left of `j` than `h`, move `j` to it and check if we have a better result.

#### C++
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

#### Python3
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
---
### 3Sum
Given an array nums of n integers, are there elements `a, b, c` in nums such that `a + b + c = 0`? Find all unique triplets in the array which gives the sum of zero.

The solution set must not contain duplicate triplets.

#### C++
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

#### Python3
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
---
### 3Sum Closest

Given an array nums of `n` integers and an integer target, find three integers in `nums` such that the sum is closest to `target`. Return the sum of the three integers. You may assume that each input would have exactly one solution.

#### C++
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

#### Python3
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
---
### 4Sum

Given an array `nums` of `n` integers and an integer `target`, are there elements `a`, `b`, `c`, and `d` in `nums` such that `a + b + c + d = target`? Find all unique quadruplets in the array which gives the sum of `target`.

The solution set must not contain duplicate quadruplets.

#### C++
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
---
### 4Sum II

Given four lists `A, B, C, D` of integer values, compute how many tuples `(i, j, k, l)` there are such that `A[i] + B[j] + C[k] + D[l]` is zero.

To make problem a bit easier, all `A, B, C, D` have same length of `N` where `0 <= N <= 500`.

#### C++
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
---
### Remove Duplicates from Sorted Array

Given a sorted array `nums`, remove the duplicates in-place such that each element appear only once and return the new length.

Do not allocate extra space for another array, you must do this by **modifying the input array in-place** with `O(1)` extra memory.


#### C++
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

---
### Remove Duplicates from Sorted Array II

Given a sorted array `nums`, remove the duplicates **in-place** such that duplicates appeared at most **twice** and return the new length.

Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.

#### Solution
At first glance, we can follow the same idea as previous problem. Compare `nums[i]` with the current last two elements of the new array. If either of the comparison return false, we can update the new array. 

In fact, we simply need to compare `nums[i] == nums[j - 2]`. If this returns false, we can update the new array no matter what.
- If `nums[i] == nums[j - 1]`, since we allow at most two duplicates, we can copy `nums[i]` to the end of the new array.

#### C++
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
---
### Find Missing Positive
Given an unsorted integer array, find the smallest missing positive integer.

Your algorithm should run in O(n) time and uses constant extra space.

#### Example
```
Input: [3,4,-1,1]
Output: 2
```

#### Solution
- Scan through `nums` and swap each positive number `A[i]` with `A[A[i]-1]`. If `A[A[i]-1]` is again positive, swap it with `A[A[A[i]-1]-1]`... Do this iteratively until we meet a negative number or we have done put all the positive numbers at their correct locations. E.g. `[3, 4, -1, 1]` will become `[1, -1, 3, 4]`.
- Iterate integers `1` to `n + 1` and check one by one if `i` is located at `i - 1` already. If not, then `i` is the first missing positive integer. 


#### Python3
```python
def firstMissingPositive(nums: 'List[int]') -> 'int':
    for i in range(len(nums)):
        while nums[i] > 0 and nums[i] <= len(nums) and nums[i] != nums[nums[i] - 1]:
            correct_idx = nums[i] - 1
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]

    for i in range(1, len(nums) + 1):
        if nums[i - 1] != i:
            return i
    return len(nums) + 1
```

---
### Insert Interval
Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).

You may assume that the intervals were initially sorted according to their start times.

#### Example
```
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
```

#### C++
```c++
vector<Interval> insert(vector<Interval> &intervals, Interval newInterval) {
    vector<Interval> result;
    for (auto it = intervals.begin(); it != intervals.end(); it++) {
        if (it->end < newInterval.start) {
            result.emplace_back(*it);
        } else if (it->start > newInterval.end) {
            result.emplace_back(newInterval);
            copy(it, intervals.end(), back_inserter(result));
            return result;
        } else {
            newInterval.start = min(newInterval.start, it->start);
            newInterval.end = max(newInterval.end, it->end);
        }
    }
    result.push_back(newInterval);
    return result;
}
```

#### Python3
```python
def insert(intervals: 'List[Interval]', newInterval: 'Interval') -> 'List[Interval]':
    s, e = newInterval.start, newInterval.end
    left_part = [_ for _ in intervals if _.end < s]
    right_part = [_ for _ in intervals if _.start > e]
    if left_part + right_part != intervals:
        s = min(s, intervals[len(left_part)].start)
        # a[~i] = a[len(a)-i-1], the i-th element from right to left
        e = max(e, intervals[~len(right_part)].end)  
    return left_part + [Interval(s, e)] + right_part
```
---
### Majority Element
Given an array of size `n`, find the majority element. The majority element is the element that appears more than `⌊ n/2 ⌋` times.

You may assume that the array is non-empty and the majority element always exist in the array.

#### Example 1
```
Input: [3,2,3]
Output: 3
```

#### Example 2
```
Input: [2,2,1,1,1,2,2]
Output: 2
```

#### C++
```c++
int majorityElement(vector<int> &nums) {
    int candidate = nums[0], count = 0;
    for (int num : nums) {
        if (count == 0) {
            candidate = num;
        }
        count += num == candidate ? 1 : -1;
    }
    return candidate;
}
```

#### Python3
```python
def majorityElement(nums: 'List[int]') -> 'int':
    count = 0
    for num in nums:
        if count == 0:
            candidate = num
        count += 1 if candidate == num else -1
    return candidate
```

---
### Majority Element II
Given an integer array of size `n`, find all elements that appear more than `⌊ n/3 ⌋` times.

Note: The algorithm should run in linear time and in O(1) space.

#### C++
```c++
struct Candidate {
    int num_, count_;

    explicit Candidate(int num, int count) : num_(num), count_(count) {}
};

vector<int> majorityElement(vector<int> &nums) {
    vector<int> result;
    if (nums.empty()) return result;
    array<Candidate, 2> candidates{Candidate(0, 0), Candidate(1, 0)};
    for (int num : nums) {
        bool flag = false;
        // If num is one of the candidates, increment its freq by 1
        for (int i = 0; i < 2; ++i) {
            if (candidates[i].num_ == num) {
                ++candidates[i].count_;
                flag = true;
                break;
            }
        }
        if (flag) continue;
        // If num is not one of the candidates and we are missing 
        // candidates, nominate it to be a new candidate
        for (int i = 0; i < 2; ++i) {
            if (candidates[i].count_ == 0) {
                candidates[i].count_ = 1;
                candidates[i].num_ = num;
                flag = true;
                break;
            }
        }
        if (flag) continue;
        // If num is not one of the candidates nor we are missing 
        // any candidates pair out current candidates by num
        for (int i = 0; i < 2; ++i) {
            --candidates[i].count_;
        }
    }
    // We now have two candidates but we still need to check
    // if both have votes more than n/3
    for (int i = 0; i < 2; ++i) {
        candidates[i].count_ = 0;
    }
    for (int num : nums) {
        for (int i = 0; i < 2; ++i) {
            if (candidates[i].num_ == num) {
                ++candidates[i].count_;
                break;
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        if (candidates[i].count_ > nums.size() / 3)
            result.emplace_back(candidates[i].num_);
    }
    return result;
}
```
---
### Kth Largest Element in an Array
Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

####Example 1:
```
Input: [3,2,1,5,6,4] and k = 2
Output: 5
```
####Example 2:
```
Input: [3,2,3,1,2,4,5,5,6] and k = 4
Output: 4
```

####Solution
When `nums.size()` is small, sort it first and return the kth element.

####Python3
```python
def findKthLargest(nums: 'List[int]', k: 'int') -> 'int':
    nums.sort(reverse=True)
    return nums[k - 1]
```

When `nums.size()` is large, use `max heap`.
####Python3
```python
import heapq

def findKthLargest(nums: 'List[int]', k: 'int') -> 'int':
    nums = [-n for n in nums];
    heapq.heapify(nums)
    for _ in range(k):
        ans = heapq.heappop(nums)
    return -ans
```

# Linked List
