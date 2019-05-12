[Github](https://github.com/MartingaleField/MartingaleField.github.io) | [Webpage](https://martingalefield.github.io/)
    
# Table of Contents

<!-- TOC depthFrom:2 -->
- [Table of Contents](#leetcode)
- [Array](#array)
    - [Two Sum](#two-sum)
    - [Container With Most Water](#container-with-most-water)
    - [3Sum](#3sum)
    - [3Sum Closest](#3sum-closest)
    - [4Sum](#4sum)
    - [4Sum II](#4sum-ii)
    - [Remove Duplicates from Sorted Array](#remove-duplicates-from-sorted-array)
    - [Remove Duplicates from Sorted Array II](#remove-duplicates-from-sorted-array-ii)
    - [Find Missing Positive](#find-missing-positive)
    - [Insert Interval](#insert-interval)
    - [Majority Element](#majority-element)
    - [Majority Element II](#majority-element-ii)
    - [Kth Largest Element in an Array](#kth-largest-element-in-an-array)
    - [Minimum Size Subarray Sum](#minimum-size-subarray-sum)
    - [Product of Array Except Self](#product-of-array-except-self)
    - [Missing Number](#missing-number)
    - [Contains Duplicate III](#contains-duplicate-iii)
    - [H-Index](#h-index)
- [Binary Search](#binary-search)
    - [H-Index II](#h-index-ii)
- [Linked List](#linked-list)
- [Binary Tree](#binary-tree)
    - [Binary Tree Inorder Traversal](#binary-tree-inorder-traversal)
    - [Binary Tree Preorder Traversal](#binary-tree-preorder-traversal)
    - [Binary Tree Postorder Traversal](#binary-tree-postorder-traversal)
    - [Binary Tree Level Order Traversal](#binary-tree-level-order-traversal)
    - [Binary Tree Zigzag Level Order Traversal](#binary-tree-zigzag-level-order-traversal)
    - [Same Tree](#same-tree)
    - [Construct Binary Tree from Preorder and Inorder Traversal](#construct-binary-tree-from-preorder-and-inorder-traversal)
    - [Construct Binary Tree from Inorder and Postorder Traversal](#construct-binary-tree-from-inorder-and-postorder-traversal)
- [Binary Search Tree](#binary-search-tree)
    - [Validate Binary Search Tree](#validate-binary-search-tree)
    - [Recover Binary Search Tree](#recover-binary-search-tree)
    - [Minimum Distance Between BST Nodes](#minimum-distance-between-bst-nodes)
- [Depth First Search](#depth-first-search)
    - [Generate Parentheses](#generate-parentheses)
    - [Sudoku Solver](#sudoku-solver)
    - [Combination Sum](#combination-sum)
    - [Combination Sum II](#combination-sum-ii)
    - [Combination Sum III](#combination-sum-iii)
- [Design](#design)
    - [LRU Cache](#lru-cache)
<!-- /TOC -->

# Array

### [Two Sum](https://leetcode.com/problems/two-sum/)

Given an array of integers, return **indices** of the two numbers such that they add up to a specific target.
You may assume that each input would have **exactly one** solution, and you may not use the same element twice.

#### Solution 

##### C++
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

##### Python3
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
[Back to Front](#table-of-contents)
---




### [Container With Most Water](https://leetcode.com/problems/container-with-most-water/)

Given `n` non-negative integers `a1, a2, ..., an` , where each represents a point at coordinate `(i, ai)`. `n` vertical lines are drawn such that the two endpoints of line `i` is at `(i, ai)` and `(i, 0)`. Find two lines, which together with x-axis forms a container, such that the container contains the most water.

![image1](https://raw.githubusercontent.com/MartingaleField/MartingaleField.github.io/master/question_11.jpg "Container With Most Water")

#### Solution 
Use two pointers. Pointer `i` points to the first element and `j` to the last. The water volume is `(j - i) * h` where `h = min(height[i], height[j])`.
* If there exists taller bar on the right of `i` than `h`, move `i` to it and check if we have a better result.
* If there exists taller bar on the left of `j` than `h`, move `j` to it and check if we have a better result.

##### C++
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

##### Python3
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
[<span style="font-size: 10px">Back to Front</span>]
---




### [3Sum](https://leetcode.com/problems/3sum/)

Given an array nums of n integers, are there elements `a, b, c` in nums such that `a + b + c = 0`? Find all unique triplets in the array which gives the sum of zero.

The solution set must not contain duplicate triplets.

#### Solution 
##### C++
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

##### Python3
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
###### [Back to Front](#table-of-contents)
---




### [3Sum Closest](https://leetcode.com/problems/3sum-closest/)

Given an array nums of `n` integers and an integer target, find three integers in `nums` such that the sum is closest to `target`. Return the sum of the three integers. You may assume that each input would have exactly one solution.

#### Solution 
##### C++
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

##### Python3
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
###### [Back to Front](#table-of-contents)
---




### [4Sum](https://leetcode.com/problems/4sum/)

Given an array `nums` of `n` integers and an integer `target`, are there elements `a`, `b`, `c`, and `d` in `nums` such that `a + b + c + d = target`? Find all unique quadruplets in the array which gives the sum of `target`.

The solution set must not contain duplicate quadruplets.

#### Solution 
##### C++
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
###### [Back to Front](#table-of-contents)
---




### [4Sum II](https://leetcode.com/problems/4sum-ii/)

Given four lists `A, B, C, D` of integer values, compute how many tuples `(i, j, k, l)` there are such that `A[i] + B[j] + C[k] + D[l]` is zero.

To make problem a bit easier, all `A, B, C, D` have same length of `N` where `0 <= N <= 500`.

#### Solution 
##### C++
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
###### [Back to Front](#table-of-contents)
---




### [Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)

Given a sorted array `nums`, remove the duplicates in-place such that each element appear only once and return the new length.

Do not allocate extra space for another array, you must do this by **modifying the input array in-place** with `O(1)` extra memory.

#### Solution 
##### C++
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
###### [Back to Front](#table-of-contents)
---




### [Remove Duplicates from Sorted Array II](https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/)

Given a sorted array `nums`, remove the duplicates **in-place** such that duplicates appeared at most **twice** and return the new length.

Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.

#### Solution
At first glance, we can follow the same idea as previous problem. Compare `nums[i]` with the current last two elements of the new array. If either of the comparison return false, we can update the new array. 

In fact, we simply need to compare `nums[i] == nums[j - 2]`. If this returns false, we can update the new array no matter what.
- If `nums[i] == nums[j - 1]`, since we allow at most two duplicates, we can copy `nums[i]` to the end of the new array.

##### C++
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
###### [Back to Front](#table-of-contents)
---




### [Find Missing Positive](https://leetcode.com/problems/first-missing-positive/)

Given an unsorted integer array, find the smallest missing positive integer.

Your algorithm should run in O(n) time and uses constant extra space.

##### Example
```
Input: [3,4,-1,1]
Output: 2
```

#### Solution
- Scan through `nums` and swap each positive number `A[i]` with `A[A[i]-1]`. If `A[A[i]-1]` is again positive, swap it with `A[A[A[i]-1]-1]`... Do this iteratively until we meet a negative number or we have done put all the positive numbers at their correct locations. E.g. `[3, 4, -1, 1]` will become `[1, -1, 3, 4]`.
- Iterate integers `1` to `n + 1` and check one by one if `i` is located at `i - 1` already. If not, then `i` is the first missing positive integer. 


##### Python3
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
###### [Back to Front](#table-of-contents)
---




### [Insert Interval](https://leetcode.com/problems/insert-interval/)

Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).

You may assume that the intervals were initially sorted according to their start times.

##### Example
```
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
```

#### Solution 
##### C++
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

##### Python3
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
###### [Back to Front](#table-of-contents)
---




### [Majority Element](https://leetcode.com/problems/majority-element/)

Given an array of size `n`, find the majority element. The majority element is the element that appears more than `⌊ n/2 ⌋` times.

You may assume that the array is non-empty and the majority element always exist in the array.

##### Example 1
```
Input: [3,2,3]
Output: 3
```

##### Example 2
```
Input: [2,2,1,1,1,2,2]
Output: 2
```

#### Solution 
##### C++
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

##### Python3
```python
def majorityElement(nums: 'List[int]') -> 'int':
    count = 0
    for num in nums:
        if count == 0:
            candidate = num
        count += 1 if candidate == num else -1
    return candidate
```
###### [Back to Front](#table-of-contents)
---




### [Majority Element II](https://leetcode.com/problems/majority-element-ii/)

Given an integer array of size `n`, find all elements that appear more than `⌊ n/3 ⌋` times.

Note: The algorithm should run in linear time and in O(1) space.

#### Solution 
##### C++
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
###### [Back to Front](#table-of-contents)
---



### [Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)

Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

##### Example 1:
```
Input: [3,2,1,5,6,4] and k = 2
Output: 5
```
##### Example 2:
```
Input: [3,2,3,1,2,4,5,5,6] and k = 4
Output: 4
```

#### Solution
When `nums.size()` is small, sort it first and return the kth element.

##### Python3
```python
def findKthLargest(nums: 'List[int]', k: 'int') -> 'int':
    nums.sort(reverse=True)
    return nums[k - 1]
```

When `nums.size()` is large, use `max heap`.
##### Python3
```python
import heapq

def findKthLargest(nums: 'List[int]', k: 'int') -> 'int':
    nums = [-n for n in nums];
    heapq.heapify(nums)
    for _ in range(k):
        ans = heapq.heappop(nums)
    return -ans
```
###### [Back to Front](#table-of-contents)
---



### [Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/)

Given an array of `n` positive integers and a positive integer `s`, find the minimal length of a **contiguous** subarray of which the `sum >= s`. If there isn't one, return `0` instead.

##### Example
```
Input: s = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: the subarray [4,3] has the minimal length under the problem constraint.
```

#### Solution 
##### C++
```c++
int minSubArrayLen(int s, vector<int> &nums) {
    int min_len = nums.size() + 1, sum = 0;
    for (int i = 0, j = 0; j < nums.size(); j++) {
        sum += nums[j];
        while (sum >= s) {
            min_len = min(min_len, j - i + 1);
            sum -= nums[i++];
        }
    }
    return min_len <= nums.size() ? min_len : 0;
}
```
###### [Back to Front](#table-of-contents)
---


### [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)

Given an array `nums` of `n` integers where `n > 1`,  return an array `output` such that `output[i]` is equal to the product of all the elements of `nums` except `nums[i]`.

Note: Please solve it without division and in O(n).

##### Example:
```
Input:  [1,2,3,4]
Output: [24,12,8,6]
```

#### Solution
- Iterate forwards over `nums` and generate `output`:
    ```
    1,  A[0],   A[0]*A[1],  ...,    A[0]*A[1]*...*A[n-3],   A[0]*A[1]*...*A[n-2]
    ```
- Iterate backwards over `nums` and update `output`:
    ```
    1 * A[1]*...*A[n-1],  A[0] * A[2]*...*A[n-1],   A[0]*A[1] * A[3]*...*A[n-1],    ...,    
    A[0]*A[1]*...*A[n-3] * A[n-1],  A[0]*A[1]*...*A[n-2] * 1
    ```
    which is the desired result.

##### Python3
```python
def productExceptSelf(nums: 'List[int]') -> 'List[int]':
    n = len(nums)
    output = [1] * n

    p = 1
    for i in range(n):
        output[i] *= p
        p *= nums[i]

    p = 1
    for i in range(n - 1, -1, -1):
        output[i] *= p
        p *= nums[i]

    return output
```
###### [Back to Front](#table-of-contents)
---



### [Missing Number](https://leetcode.com/problems/missing-number/)
Given an array containing n distinct numbers taken from `0, 1, 2, ..., n`, find the one that is missing from the array.

Your algorithm should run in linear runtime complexity. Could you implement it using only constant extra space complexity?

##### Example 1:
```
Input: [3,0,1]
Output: 2
```
##### Example 2:
```
Input: [9,6,4,2,3,5,7,0,1]
Output: 8
```

#### Solution: Math
The missing one is `sum(0..n) - sum(nums)`.
##### Python3
```python
def missingNumber(nums: 'List[int]') -> 'int':
    n = len(nums)
    return (n * (n + 1) // 2) - sum(nums)
```

#### Solution: Bit Manipulation
If `A == B`, then `A ^ B == 0`.

##### Python3
```python
def missingNumber(nums: 'List[int]') -> 'int':
    ans = len(nums)
    for i in range(0, len(nums)):
        ans ^= (nums[i] ^ i)
    return ans
```
###### [Back to Front](#table-of-contents)
---

### [Contains Duplicate III](https://leetcode.com/problems/contains-duplicate-iii/)

Given an array of integers, find out whether there are two distinct indices `i` and `j` in the array such that the **absolute** difference between `nums[i]` and `nums[j]` is at most `t` and the **absolute** difference between `i` and `j` is at most `k`.

##### Example 1:
```
Input: nums = [1,2,3,1], k = 3, t = 0
Output: true
```
##### Example 2:
```
Input: nums = [1,0,1,1], k = 1, t = 2
Output: true
```
##### Example 3:
```
Input: nums = [1,5,9,1,5,9], k = 2, t = 3
Output: false
```

#### Solution: Sort
Use a `vector<pair<long, int>>` to store `(elem, index)` pairs. Sort this vector. This will produce a similar structure to `multimap<long, int>` but we can do sliding-window technique on it using continuous indexing.

##### C++
```c++
bool containsNearbyAlmostDuplicate(vector<int> &nums, int k, int t) {
    vector<pair<long, int>> map;
    for (int i = 0; i < nums.size(); ++i)
        map.push_back({nums[i], i});
    sort(map.begin(), map.end());
    int j = 1;
    for (int i = 0; i < map.size(); ++i) {
        while (j < map.size() && abs(map[j].first - map[i].first) <= t) {
            if (abs(map[j].second - map[i].second) <= k)
                return true;
            j++;
        }
        if (j == i + 1) j++;
    }
    return false;
}
```

##### Python3
```python
def containsNearbyAlmostDuplicate(nums: List[int], k: int, t: int) -> bool:
    map = [(e, i) for i, e in enumerate(nums)]
    map.sort()
    j = 1
    for i in range(len(map)):
        while j < len(map) and abs(map[j][0] - map[i][0]) <= t:
            if abs(map[i][1] - map[j][1]) <= k:
                print(i, j)
                return True
            j += 1
        if j == i + 1:
            j += 1
    return False
```

#### Solution: Ordered Set

The sliding-window idea can also be implemented using `set<long>`, in which elements are ordered automatically.

##### C++
```c++
bool containsNearbyAlmostDuplicate(vector<int> &nums, int k, int t) {
    set<long> window; // set is ordered automatically
    for (int i = 0; i < nums.size(); i++) {
        // keep the set contains nums with |i - j| at most k
        if (i > k) window.erase(nums[i - k - 1]);
        // |x - nums[i]| <= t  ==> -t <= x - nums[i] <= t;
        auto pos = window.lower_bound(static_cast<long>(nums[i]) - t); // x - nums[i] >= -t ==> x >= nums[i]-t
        // x - nums[i] <= t ==> |x - nums[i]| <= t
        if (pos != window.end() && *pos - nums[i] <= t) return true;
        window.insert(nums[i]);
    }
    return false;
}
```
###### [Back to Front](#table-of-contents)
---


### [H-Index](https://leetcode.com/problems/h-index/)

Given an array of citations (each citation is a non-negative integer) of a researcher, write a function to compute the researcher's h-index.

According to the definition of h-index on Wikipedia: "A scientist has index h if h of his/her N papers have at least h citations each, and the other N − h papers have no more than h citations each."

Example:
```
Input: citations = [3,0,6,1,5]
Output: 3 
Explanation: [3,0,6,1,5] means the researcher has 5 papers in total and each of them had 
             received 3, 0, 6, 1, 5 citations respectively. 
             Since the researcher has 3 papers with at least 3 citations each and the remaining 
             two with no more than 3 citations each, her h-index is 3.
```
Note: If there are several possible values for h, the maximum one is taken as the h-index.

#### Solution

![h-index](https://upload.wikimedia.org/wikipedia/commons/d/da/H-index-en.svg)

If we sort the `citations` in decreasing order, the h-index is then the last position where the citation is **greater than** the position. 

For example, for input `[3,3,3,0,6,1,5]` (`[6,5,3,3,3,1,0]` after sorting), the last position where the citation is greater than the position is 3 (the position starts from 1). Hence the h-index is 3.

Algorithm:

- Counting sort: Take a `cnt` array of size `N + 1`. If a paper has a citation of `c <= N`, `cnt[c]++`; if `c > N`, `cnt[N]++`. The reason for the second `if` is that the h-index cannot be larger than `N` and so we can treat all citations larger than `N` the same. 

- We then scan from right to left, summing up `cnt[i]` along the way, until we reach a sum greater than or equal to the current index. Then this index is our h-index.

A simple implementation using built-in `sort` function can be

##### C++
```c++
int hIndex(vector<int> &citations) {
    sort(citations.begin(), citations.end(), greater<int>());
    int h = 0;
    for (int i = 0; i < citations.size(); ++i)
        if (citations[i] > i) ++h;
    return h;
}
```
##### Python3
```python
def hIndex(citations: List[int]) -> int:
    h = 0
    for i, c in enumerate(sorted(citations, reverse=True)):
        if c > i:
            h += 1
    return h
```

But this has a complexity of `O(n logn)`, so is applicable if `n` is small. When `n` is large, we use counting sort.

##### C++
```c++
int hIndex(vector<int> &citations) {
    int n = citations.size();
    vector<int> cnt(n + 1, 0);
    for (int c : citations) {
        if (c >= n)
            cnt[n]++;
        else
            cnt[c]++;
    }
    int sum = 0;
    for (int i = n; i >= 0; --i) {
        sum += cnt[i];
        if (sum >= i) return i;
    }
    return 0;
}
```

##### Python3
```python
def hIndex(self, citations: List[int]) -> int:
    n = len(citations)
    cnt = [0] * (n + 1);
    for c in citations:
        if c <= n:
            cnt[c] += 1
        else:
            cnt[n] += 1

    ans = 0
    for i in range(n, -1, -1):
        ans += cnt[i]
        if ans >= i:
            return i
    return 0
```

###### [Back to Front](#table-of-contents)
---


# Binary Search

### [H-Index II](https://leetcode.com/problems/h-index-ii/)

Given an array of citations **sorted in ascending order** (each citation is a non-negative integer) of a researcher, write a function to compute the researcher's h-index.

If there are several possible values for h, the maximum one is taken as the h-index.

#### Solution

The idea is to search for the first index so that
```
citations[index] >= length(citations) - index
```
And return `length - index` as the result. The search can be done using binary search.

##### C++
```c++
int hIndex(vector<int> &citations) {
    int n = citations.size();
    int left = 0, right = n - 1;
    while (left <= right) {
        int mid = (left + right) >> 1;
        if (citations[mid] == n - mid)
            return n - mid;
        else if (citations[mid] < n - mid)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return n - left;
}
```

###### [Back to Front](#table-of-contents)
---




# Linked List


# Binary Tree

### [Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)

Given a binary tree, return the inorder traversal of its nodes' values.

##### Example
```
Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [1,3,2]
```

#### Solution: Recursive

##### C++
```c++
class Solution {
public:
    vector<int> inorderTraversal(TreeNode *root) {
        inorder(root);
        return result;
    }

private:
    vector<int> result;

    void inorder(TreeNode *root) {
        if (root == nullptr) return;
        
        inorder(root->left);
        result.push_back(root->val);
        inorder(root->right);
    }
};
```

#### Solution: Iterative

##### C++
```c++
vector<int> inorderTraversal(TreeNode *root) {
    vector<int> result;
    stack<TreeNode *> s; // nodes to be visited
    auto node = root;
    while (!s.empty() || node != nullptr) {
        if (node != nullptr) {
            s.push(node);
            node = node->left;
        } else {
            node = s.top();
            s.pop();
            result.emplace_back(node->val);
            node = node->right;
        }
    }
    return result;
}
```

#### Solution: Morris

A binary tree is threaded by making all right child pointers that would normally be null point to the inorder successor of the node (if it exists), and all left child pointers that would normally be null point to the inorder predecessor of the node.

![image3](https://upload.wikimedia.org/wikipedia/commons/7/7a/Threaded_tree.svg "Threaded Binary Tree")

Inorder: ABCDEFGHI

The threads we need for inorder traveral are `A->B`, `C->D`, `E->F` and `H->I`. At each subtree, we first thread `p` to `cur` (root of subtree) and next time we print out `p` we can use this thread to visit and print out `cur`. 

##### Pseudo Code
```
1. Initialize current as root 
2. While current is not NULL
   If current does not have a left child
      ia) Print current’s data
      ib) Go to the right, i.e., current = current->right
   Else
      ea) Make current as right child of the rightmost node in current's left subtree
      eb) Go to this left child, i.e., current = current->left
```

Time complexity O(n), space complexity O(1).

##### C++
```c++
vector<int> inorderTraversal(TreeNode *root) {
    vector<int> result;
    TreeNode *cur = root, *p = nullptr;
    while (cur != nullptr) {
        if (cur->left == nullptr) { // cur has no left child
            result.emplace_back(cur->val);
            cur = cur->right;
        } else { // cur has left child
            // Let p point to the rightmost node of cur->left
            for (p = cur->left; p->right != nullptr && p->right != cur; p = p->right);

            if (p->right == nullptr) { // p has not been threaded to cur
                p->right = cur; 
                cur = cur->left;
            } else { // p is already threaded to cur                
                result.emplace_back(cur->val); // This line is different from preorder traversal
                p->right = nullptr; // remove thread
                cur = cur->right;
            }
        }
    }
    return result;
}
```
###### [Back to Front](#table-of-contents)
---

### [Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)

Given a binary tree, return the preorder traversal of its nodes' values.

##### Example
```
Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [1,2,3]
```

#### Solution: Recursive

##### C++
```c++
class Solution {
public:
    vector<int> preorderTraversal(TreeNode *root) {
        preorder(root);
        return result;
    }

private:
    vector<int> result;

    void preorder(TreeNode *root) {
        if (root == nullptr) return;

        result.push_back(root->val);
        preorder(root->left);
        preorder(root->right);
    }
};
```

#### Solution: Iterative

##### C++
```c++
vector<int> preorderTraversal(TreeNode *root) {
    vector<int> result;
    stack<TreeNode *> s; // nodes that have been visited
    auto node = root;
    while (!s.empty() || node != nullptr) {
        if (node != nullptr) {
            s.push(node);
            result.emplace_back(node->val);
            node = node->left;
        } else {
            node = s.top();
            s.pop();
            node = node->right;
        }
    }
    return result;
}
```

#### Solution: Morris

![image3](https://upload.wikimedia.org/wikipedia/commons/7/7a/Threaded_tree.svg "Threaded Binary Tree")

Preorder: FBADCEGIH

The threads we need for preorder traveral are `A->B`, `C->D`, `E->F` and `H->I`. The difference with inorder is that we print out `cur` before threading `p` to `cur`. The reason is that in preorder traversal, we need to visit the root first before traversing the left subtree.

##### C++
```c++
vector<int> preorderTraversal(TreeNode *root) {
    vector<int> result;
    TreeNode *cur = root, *p = nullptr;
    while (cur != nullptr) {
        if (cur->left == nullptr) { // cur has no left child
            result.emplace_back(cur->val);
            cur = cur->right;
        } else { // cur has left child
            // Let p point to the rightmost node of cur->left
            for (p = cur->left; p->right != nullptr && p->right != cur; p = p->right);

            if (p->right == nullptr) { // p has not been threaded to cur
                result.emplace_back(cur->val); // This line is different from inorder traversal
                p->right = cur;
                cur = cur->left;
            } else { // p is already threaded to cur
                p->right = nullptr; // remove thread
                cur = cur->right;
            }
        }
    }
    return result;
}
```
###### [Back to Front](#table-of-contents)
---

### [Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/)

Given a binary tree, return the postorder traversal of its nodes' values.

#### Solution: Recursive

##### C++
```c++
class Solution {
public:
    vector<int> postorderTraversal(TreeNode *root) {
        preorder(root);
        return result;
    }

private:
    vector<int> result;

    void preorder(TreeNode *root) {
        if (!root) return;

        preorder(root->left);
        preorder(root->right);
        result.push_back(root->val);
    }
};
```

#### Solution: Iterative

##### C++
```c++
vector<int> postorderTraversal(TreeNode *root) {
    vector<int> result;
    stack<TreeNode *> s; // nodes to be visited
    auto node = root;
    while (!s.empty() || node != nullptr) {
        if (node != nullptr) {
            s.push(node);
            result.push_back(node->val);
            node = node->right; // node = node->left for preorder
        } else {
            node = s.top();
            s.pop();
            node = node->left; // node = node->right for preorder
        }
    }
    reverse(result.begin(), result.end()); // reverse in the end
    return result;
}
```

#### Solution: Morris

##### C++
```c++
void reverse(TreeNode *from, TreeNode *to) {
    TreeNode *x = from, *y = from->right, *z;
    if (from == to) return;
    while (x != to) {
        z = y->right;
        y->right = x;
        x = y;
        y = z;
    }
}

template<typename func>
void visit_reverse(TreeNode *from, TreeNode *to, func &visit) {
    auto p = to;
    reverse(from, to);

    while (true) {
        visit(p);
        if (p == from) break;
        p = p->right;
    }
    reverse(to, from);
}

vector<int> postorderTraversal(TreeNode *root) {
    vector<int> result;
    TreeNode dummy(-1);
    dummy.left = root;

    auto visit = [&result](TreeNode *node) { result.emplace_back(node->val); };

    TreeNode *cur = &dummy, *prev = nullptr, *p = nullptr;
    while (cur != nullptr) {
        if (cur->left == nullptr) {
            prev = cur;
            cur = cur->right;
        } else {
            for (p = cur->left; p->right != nullptr && p->right != cur; p = p->right);

            if (p->right == nullptr) {
                p->right = cur;
                prev = cur;
                cur = cur->left;
            } else {
                visit_reverse(cur->left, prev, visit);
                prev->right = nullptr;
                prev = cur;
                cur = cur->right;
            }
        }
    }
    return result;
}
```

###### [Back to Front](#table-of-contents)
---

### [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)

Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

##### Example
```
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its level order traversal as:
[
  [3],
  [9,20],
  [15,7]
]
```

#### Solution: Recursive
##### C++
```c++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode *root) {
        traverse(root, 0);
        return result;
    }

private:
    vector<vector<int>> result;

    void traverse(TreeNode *root, int level) {
        if (root == nullptr) return;

        if (level == result.size())
            result.emplace_back(vector<int>{});

        result[level].emplace_back(root->val);
        traverse(root->left, level + 1);
        traverse(root->right, level + 1);
    }
};
```

#### Solution: Iterative

Do an **preorder** traversal using the interative way. Push each node and its level in the stack.

##### C++
```c++
vector<vector<int>> levelOrder(TreeNode *root) {
    vector<vector<int>> result;
    stack<pair<TreeNode *, int>> s;
    auto node = root;
    int level = -1;
    while (!s.empty() || node != nullptr) {
        if (node != nullptr) {
            s.push({node, ++level});
            if (level == result.size())
                result.emplace_back(vector<int>{});
            result[level].emplace_back(node->val);
            node = node->left;
        } else {
            node = s.top().first;
            level = s.top().second;
            s.pop();
            node = node->right;
        }
    }
    return result;
}
```

#### Solution: Queue

##### C++
```c++
vector<vector<int>> levelOrder(TreeNode *root) {
    if (!root) return {};

    vector<vector<int>> result;
    queue<TreeNode *> current, next;
    current.emplace(root);
    while (!current.empty()) {
        vector<int> level;
        while (!current.empty()) {
            auto node = current.front();
            current.pop();
            level.emplace_back(node->val);
            if (node->left) next.emplace(node->left);
            if (node->right) next.emplace(node->right);
        }
        result.emplace_back(level);
        swap(next, current);
    }
    return result;
}
```

###### [Back to Front](#table-of-contents)
---

### [Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)

Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).

For example:
Given binary tree `[3,9,20,null,null,15,7]`,
```
    3
   / \
  9  20
    /  \
   15   7
```
return its zigzag level order traversal as:
```
[
  [3],
  [20,9],
  [15,7]
]
```

#### Solution: Recursive

##### C++
```c++
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode *root) {
        traverse(root, 0);
        return result;
    }

private:
    vector<vector<int>> result;

    void traverse(TreeNode *root, int level) {
        if (root == nullptr) return;

        if (level == result.size())
            result.push_back({});

        if (level % 2 == 0) // traverse left to right if row index is even
            result[level].push_back(root->val);
        else                // traverse right to left if row index is odd
            result[level].insert(result[level].begin(), root->val);

        traverse(root->left, level + 1);
        traverse(root->right, level + 1);
    }
};
```

#### Solution: Iterative, Stack

##### C++
```c++
vector<vector<int>> zigzagLevelOrder(TreeNode *root) {
    vector<vector<int>> result;
    stack<pair<TreeNode *, int>> s;
    auto node = root;
    int level = -1;
    bool isLR = true;
    while (!s.empty() || node != nullptr) {
        if (node != nullptr) {
            s.push({node, ++level});
            if (level == result.size())
                result.push_back({});
            if (level % 2 == 0)
                result[level].push_back(node->val);
            else
                result[level].insert(result[level].begin(), node->val);
            node = node->left;
        } else {
            node = s.top().first;
            level = s.top().second;
            s.pop();
            node = node->right;
        }
    }
    return result;
}
```

###### [Back to Front](#table-of-contents)
---





### [Same Tree](https://leetcode.com/problems/same-tree/)

Given two binary trees, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical and the nodes have the same value.

#### Solution: Recursive

##### C++
```c++
bool isSameTree(TreeNode *p, TreeNode *q) {
    if (!p && !q) return true;
    if (!p || !q) return false;
    return p->val == q->val
           && isSameTree(p->left, q->left)
           && isSameTree(p->right, q->right);
}
```

#### Solution: Iterative

##### C++
```c++
bool isSymmetric(TreeNode *root) {
    if (!root) return true;

    stack<TreeNode *> s;
    s.push(root->left);
    s.push(root->right);

    while (!s.empty()) {
        auto p = s.top();
        s.pop();
        auto q = s.top();
        s.pop();

        if (!p && !q) continue;
        if (!p || !q) return false;
        if (p->val != q->val) return false;

        s.push(p->right);
        s.push(q->left);
        s.push(p->left);
        s.push(q->right);
    }
    return true;
}
```

###### [Back to Front](#table-of-contents)
---



### [Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

Given preorder and inorder traversal of a tree, construct the binary tree.

You may assume that duplicates do not exist in the tree.

For example, given
```
preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
```
Return the following binary tree:
```
    3
   / \
  9  20
    /  \
   15   7
```

#### Solution

Inorder:
```
[ left subtree ] root [ right subtree ]
```
Preorder:
```
root [ left subtree ] [ right subtree ]
```
We first find the position of root (i.e. `*begin(preorder)`) in inorder vector. The size of left subtree is then `distance(begin(inorder), in_root_pos)`. Then recursively build left and right subtrees.
- For left subtree, the inorder vector is
    `inorder[0..left_size - 1]`
    the preorder vector is
    `preorder[1..left_size]`
- For right subtree, the inorder vector is
    `inorder[in_root_pos + 1..in_last]`
    the preorder vector is
    `preorder[left_size + 1..pre_last]`

##### C++
```c++
template<typename InputIterator>
TreeNode *buildTree(InputIterator pre_first, InputIterator pre_last,
                    InputIterator in_first, InputIterator in_last) {
    if (pre_first == pre_last) return nullptr;
    if (in_first == in_last) return nullptr;

    auto root = new TreeNode(*pre_first);
    auto in_root_pos = find(in_first, in_last, *pre_first);
    auto left_size = distance(in_first, in_root_pos);

    root->left = buildTree(next(pre_first), next(pre_first, left_size + 1),
                           in_first, next(in_first, left_size));
    root->right = buildTree(next(pre_first, left_size + 1),
                            pre_last, next(in_root_pos), in_last);

    return root;
}

TreeNode *buildTree(vector<int> &preorder, vector<int> &inorder) {
    return buildTree(begin(preorder), end(preorder), begin(inorder), end(inorder));
}
```
###### [Back to Front](#table-of-contents)
---


### [Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

Given inorder and postorder traversal of a tree, construct the binary tree.

You may assume that duplicates do not exist in the tree.

For example, given
```
inorder = [9,3,15,20,7]
postorder = [9,15,7,20,3]
```
Return the following binary tree:
```
    3
   / \
  9  20
    /  \
   15   7
```

#### Solution

Inorder:
```
[ left subtree ] root [ right subtree ]
```
Postorder:
```
[ left subtree ] [ right subtree ] root
```

##### C++
```c++
template<typename InputIterator>
TreeNode *buildTree(InputIterator in_first, InputIterator in_last,
                    InputIterator post_first, InputIterator post_last) {
    if (in_first == in_last) return nullptr;
    if (post_first == post_last) return nullptr;

    auto root_val = *prev(post_last);
    TreeNode *root = new TreeNode(root_val);

    auto in_root_pos = find(in_first, in_last, root_val);
    auto left_size = distance(in_first, in_root_pos);

    root->left = buildTree(in_first, in_root_pos,
                           post_first, next(post_first, left_size));
    root->right = buildTree(next(in_root_pos), in_last,
                            next(post_first, left_size), prev(post_last));
    return root;
}

TreeNode *buildTree(vector<int> &inorder, vector<int> &postorder) {
    return buildTree(begin(inorder), end(inorder), begin(postorder), end(postorder));
}
```

###### [Back to Front](#table-of-contents)
---




# Binary Search Tree

### [Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)

Given a binary tree, determine if it is a valid binary search tree (BST).

Assume a BST is defined as follows:
- The left subtree of a node contains only nodes with keys **less than** the node's key.
- The right subtree of a node contains only nodes with keys **greater than** the node's key.
- Both the left and right subtrees must also be binary search trees.

#### Solution: Recursive

##### C++
```c++
bool helper(TreeNode *node, TreeNode *min_node = nullptr, TreeNode *max_node = nullptr) {
    if (node == nullptr) return true;

    if ((min_node != nullptr && node->val <= min_node->val) ||
        (max_node != nullptr && node->val >= max_node->val))
        return false;

    return helper(node->right, node, max_node) && helper(node->left, min_node, node);
}

bool isValidBST(TreeNode *root) {
    return helper(root);
}
```

#### Solution: Iterative

Do an inorder traversal and compare `node->val` with `pre-val` along the way.

##### C++
```c++
bool isValidBST(TreeNode *root) {
    vector<int> result;
    stack<TreeNode *> s; // nodes to be visited
    TreeNode *pre = nullptr, *node = root;
    while (!s.empty() || node != nullptr) {
        if (node != nullptr) {
            s.push(node);
            node = node->left;
        } else {
            node = s.top();
            s.pop();
            if (pre != nullptr && node->val <= pre->val)
                return false;
            pre = node;
            node = node->right;
        }
    }
    return true;
}
```

###### [Back to Front](#table-of-contents)
---




### [Recover Binary Search Tree](https://leetcode.com/problems/recover-binary-search-tree/)

Two elements of a binary search tree (BST) are swapped by mistake.

Recover the tree without changing its structure.

##### Example 1:
```
Input: [1,3,null,null,2]

   1
  /
 3
  \
   2

Output: [3,1,null,null,2]

   3
  /
 1
  \
   2
```

##### Example 2:
```
Input: [3,1,4,null,null,2]

  3
 / \
1   4
   /
  2

Output: [2,1,4,null,null,3]

  2
 / \
1   4
   /
  3
```

#### Solution: Straightforward, O(n) space

We can use a `vector<TreeNode *> inorder` to store the inorder-traversed nodes. This can be implemented by any one of the inorder traversal approaches (recursive, stack-based iterative or Morris). If the BST is valid, `inorder` should be non-decreasing. We can then forwards-iterate the vector and find the node `broken1` which violates the ordering. Similarly, we can backwards-iterate the vector and find the node `broken2` which violates the ordering. Swapping `broken1` and `broken2` yields the valid BST.

##### C++
```c++
void recoverTree(TreeNode *root) {
    vector<TreeNode *> inorder;
    stack<TreeNode *> s;
    auto cur = root, broken1 = root, broken2 = root;
    while (!s.empty() || cur) {
        if (cur) {
            s.push(cur);
            cur = cur->left;
        } else {
            cur = s.top();
            s.pop();
            inorder.emplace_back(cur);
            cur = cur->right;
        }
    }
    for (int i = 0; i < inorder.size() - 1; ++i) {
        if (inorder[i]->val > inorder[i + 1]->val) {
            broken1 = inorder[i];
            break;
        }
    }
    for (int i = inorder.size() - 1; i > 0; --i) {
        if (inorder[i]->val < inorder[i - 1]->val) {
            broken2 = inorder[i];
            break;
        }
    }
    swap(broken1->val, broken2->val);
}
```

#### Solution: Iterative, Stack

##### C++
```c++
class Solution {
public:
    void recoverTree(TreeNode *root) {
        stack<TreeNode *> s;
        TreeNode *pre = nullptr, *cur = root;
        while (!s.empty() || cur != nullptr) {
            if (cur != nullptr) {
                s.push(cur);
                cur = cur->left;
            } else {
                cur = s.top();
                s.pop();
                detect(pre, cur);
                pre = cur;
                cur = cur->right;
            }
        }
        swap(broken1->val, broken2->val);
    }

private:
    TreeNode *broken1 = nullptr, *broken2 = nullptr;

    void detect(TreeNode *prev, TreeNode *curr) {
        if (prev != nullptr && prev->val > curr->val) {
            if (broken1 == nullptr)
                broken1 = prev;
            broken2 = curr;
        }
    }
};
```

#### Solution: Recursive, O(1) space

Actually, we don't need to record all inorder-traversed nodes. We simply need a `TreeNode *pre` which points to the inorder predecessor of the currently visiting node.

##### C++
```c++
class Solution {
public:
    void recoverTree(TreeNode *root) {
        inorder(root);
        swap(broken1->val, broken2->val);
    }

private:
    TreeNode *broken1 = nullptr, *broken2 = nullptr, *pre = nullptr;

    void detect(TreeNode *prev, TreeNode *curr) {
        if (prev != nullptr && prev->val > curr->val) {
            if (broken1 == nullptr)
                broken1 = prev;
            broken2 = curr;
        }
    }

    void inorder(TreeNode *cur) {
        if (cur == nullptr) return;

        inorder(cur->left);
        detect(pre, cur);
        pre = cur;
        inorder(cur->right);
    }
};
```


#### Solution: Morris

##### C++
```c++
class Solution {
public:
    void recoverTree(TreeNode *root) {
        TreeNode *prev = nullptr, *cur = root, *p = nullptr;
        while (cur != nullptr) {
            if (cur->left == nullptr) {
                detect(prev, cur); // "visit"
                prev = cur;
                cur = cur->right;
            } else {
                for (p = cur->left; p->right != nullptr && p->right != cur; p = p->right);

                if (p->right == nullptr) {
                    p->right = cur;
                    cur = cur->left;
                } else {
                    detect(prev, cur); // "visit"
                    prev = cur;
                    p->right = nullptr;
                    cur = cur->right;
                }
            }
        }
        swap(broken1->val, broken2->val);
    }

private:
    TreeNode *broken1 = nullptr, *broken2 = nullptr;

    void detect(TreeNode *prev, TreeNode *curr) {
        if (prev != nullptr && prev->val > curr->val) {
            if (broken1 == nullptr)
                broken1 = prev;
            broken2 = curr;
        }
    }
};
```
###### [Back to Front](#table-of-contents)
---


### [Minimum Distance Between BST Nodes](https://leetcode.com/problems/minimum-distance-between-bst-nodes/)

Given a Binary Search Tree (BST) with the root node `root`, return the minimum difference between the values of any two different nodes in the tree.

Example:
```
Input: root = [4,2,6,1,3,null,null]
Output: 1
Explanation:
Note that root is a TreeNode object, not an array.

The given tree [4,2,6,1,3,null,null] is represented by the following diagram:

          4
        /   \
      2      6
     / \    
    1   3  

while the minimum difference in this tree is 1, it occurs between node 1 and node 2, also between node 3 and node 2.
```

#### Solution: Recursive

Do an inorder traversal and record predecessor node `pre` along the way. The minimum distance can only happen between two consecutive nodes in the traversal.

##### C++
```c++
class Solution {
public:
    int minDiffInBST(TreeNode *root) {
        return inorder(root);
    }

private:
    TreeNode *pre = nullptr;

    int inorder(TreeNode *cur) {
        if (cur == nullptr) return INT_MAX;

        int min_dist = inorder(cur->left);
        if (pre != nullptr)
            min_dist = min(min_dist, cur->val - pre->val);
        pre = cur;
        return min(min_dist, inorder(cur->right));
    }
};
```
###### [Back to Front](#table-of-contents)
---


# Depth First Search

### [Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)

Given `n` pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

For example, given `n = 3`, a solution set is:
```
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
```

#### Solution: DFS

##### C++
```c++
class Solution {
public:
    vector<string> generateParenthesis(int n) {
        helper(n, n, "");
        return result;
    }

private:
    vector<string> result;

    void helper(int n_left, int n_right, string valid_prefix) {
        if (!n_right) {
            result.emplace_back(valid_prefix);
            return;
        }
        if (n_left > 0) 
            helper(n_left - 1, n_right, valid_prefix + '(');
        if (n_right > n_left) 
            helper(n_left, n_right - 1, valid_prefix + ')');
    }
};
```

##### Python3
```python
def generateParenthesis(n: 'int') -> 'List[str]':
    def backtrack(s, l, r):
        if l == 0 and r == 0:
            ans.append(s)
            return
        if l > 0:
            backtrack(s + '(', l - 1, r)
        if r > l:
            backtrack(s + ')', l, r - 1)

    ans = []
    backtrack("", n, n)
    return ans
```
###### [Back to Front](#table-of-contents)
---


### [Sudoku Solver](https://leetcode.com/problems/sudoku-solver/)

Write a program to solve a Sudoku puzzle by filling the empty cells.

A sudoku solution must satisfy all of the following rules:

- Each of the digits 1-9 must occur exactly once in each row.
- Each of the digits 1-9 must occur exactly once in each column.
- Each of the the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
- Empty cells are indicated by the character `'.'`.

You may assume that the given Sudoku puzzle will have a single unique solution.

![image](https://upload.wikimedia.org/wikipedia/commons/e/e0/Sudoku_Puzzle_by_L2G-20050714_standardized_layout.svg "A typical Sudoku puzzle")
![image](https://upload.wikimedia.org/wikipedia/commons/1/12/Sudoku_Puzzle_by_L2G-20050714_solution_standardized_layout.svg "solution")

#### Solution: Backtracking

##### C++
```c++
class Solution {
public:
    void solveSudoku(vector<vector<char>> &board) {
        if (board.size() < 9 || board[0].size() < 9)
            return;
        bool res = dfs(board, 0, 0);
    }

private:
    bool dfs(vector<vector<char>> &board, int i, int j) {
        if (i == 9) return true;

        int i2 = (i + (j + 1) / 9), j2 = (j + 1) % 9;
        if (board[i][j] != '.') {
            if (!isValid(board, i, j))
                return false;
            return dfs(board, i2, j2);
        } else {
            for (int k = 0; k < 9; k++) {
                board[i][j] = '1' + k;
                if (isValid(board, i, j) && dfs(board, i2, j2))
                    return true;
            }
            board[i][j] = '.';
            return false;
        }
    }

    bool isValid(vector<vector<char>> &board, int x, int y) {
        int i, j;
        for (i = 0; i < 9; ++i) 
            if (i != x && board[i][y] == board[x][y])
                return false;

        for (j = 0; j < 9; ++j) 
            if (j != y && board[x][j] == board[x][y])
                return false;

        for (i = 3 * (x / 3); i < 3 * (x / 3 + 1); ++i) 
            for (j = 3 * (y / 3); j < 3 * (y / 3 + 1); ++j) 
                if ((i != x || j != y) && board[i][j] == board[x][y])
                    return false;
        return true;
    }
};
```

#### Solution: Backtracking with Caching

##### C++
```c++
class Solution {
public:
    // row[i][j], column[i][j], subcube[i][j] represents repectively
    // if row/column/subcube i (1..9) has number j (1..9)
    // combine them into one bitset with size 9 * 9 * 3
    bitset<9 * 9 * 3> flag;

    void solveSudoku(vector<vector<char>> &board) {
        if (board.size() < 9) return;

        flag.reset();
        for (uint8_t i = 0; i < 9; i++) {
            for (uint8_t j = 0; j < 9; j++) {
                if (board[i][j] == '.') continue;

                auto num = static_cast<uint8_t>(board[i][j] - '1');
                auto cube = static_cast<uint8_t>(i / 3 * 3 + j / 3);
                auto row_num = static_cast<uint8_t>(i * 9 + num);
                auto col_num = static_cast<uint8_t>(j * 9 + num + 81);
                auto cb_num = static_cast<uint8_t>(cube * 9 + num + 81 * 2);
                if (flag[row_num] || flag[col_num] || flag[cb_num])
                    return;
                flag.set(row_num);
                flag.set(col_num);
                flag.set(cb_num);
            }
        }
        step(board, 0, 0);
    }

    bool step(vector<vector<char>> &board, uint8_t i, uint8_t j) {
        if (i == 9) return true;

        auto i2 = static_cast<uint8_t>(i + (j + 1) / 9);
        auto j2 = static_cast<uint8_t>((j + 1) % 9);
        if (board[i][j] != '.') {
            if (i == 8 && j == 8) 
                return true;
            else 
                return step(board, i2, j2);
        }
        auto cube = static_cast<uint8_t>(i / 3 * 3 + j / 3);
        for (uint8_t k = 0; k < 9; k++) {
            auto row_num = static_cast<uint8_t>(i * 9 + k);
            auto col_num = static_cast<uint8_t>(j * 9 + k + 81);
            auto cb_num = static_cast<uint8_t>(cube * 9 + k + 81 * 2);
            if (flag[row_num] || flag[col_num] || flag[cb_num])
                continue;
            flag.set(row_num);
            flag.set(col_num);
            flag.set(cb_num);
            board[i][j] = '1' + k;

            if (step(board, i2, j2))
                return true;
            flag.reset(row_num);
            flag.reset(col_num);
            flag.reset(cb_num);
            board[i][j] = '.';
        }
        return false;
    }
};
```
###### [Back to Front](#table-of-contents)
---


### [Combination Sum](https://leetcode.com/problems/combination-sum/)

Given a set of candidate numbers (`candidates`) (without duplicates) and a target number (`target`), find all unique combinations in `candidates` where the candidate numbers sums to `target`.

The same repeated number may be chosen from `candidates` unlimited number of times.

Note:
- All numbers (including `target`) will be positive integers.
- The solution set must not contain duplicate combinations.

Example 1:
```c++
Input: candidates = [2,3,6,7], target = 7,
A solution set is:
[
  [7],
  [2,2,3]
]
```
Example 2:
```c++
Input: candidates = [2,3,5], target = 8,
A solution set is:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
```

#### Solution: DFS
##### C++
```c++
class Solution {
public:
    vector<vector<int>> combinationSum(vector<int> &candidates, int target) {
        sort(candidates.begin(), candidates.end());
        dfs(candidates, target, 0);
        return result;
    }

private:
    vector<vector<int>> result;
    vector<int> path;

    void dfs(vector<int> &candidates, int gap, int cur) {
        if (!gap) {
            result.push_back(path);
            return;
        }
        
        for (int i = cur; i < candidates.size(); ++i) {
            if (gap < candidates[i]) break;
            path.emplace_back(candidates[i]);
            // next step starts from i because duplicates are allowed
            dfs(candidates, gap - candidates[i], i);
            path.pop_back();
        }
    }
};
```

##### Python3
```python
def combinationSum(candidates: 'List[int]', target: 'int') -> 'List[List[int]]':
    def dfs(gap, cur):
        if gap == 0:
            result.append(path[:])
            return
        for i in range(cur, len(candidates)):
            if gap < candidates[i]:
                break
            path.append(candidates[i])
            # next step starts from i because duplicates are allowed
            dfs(gap - candidates[i], i)
            path.pop()

    result = []
    path = []
    candidates.sort()
    dfs(target, 0)
    return result
```
###### [Back to Front](#table-of-contents)
---



### [Combination Sum II](https://leetcode.com/problems/combination-sum-ii/)

Given a collection of candidate numbers (`candidates`) and a target number (`target`), find all unique combinations in `candidates` where the candidate numbers sums to `target`.

Each number in `candidates` may only be used **once** in the combination.

Note:
- All numbers (including `target`) will be positive integers.
- The solution set must not contain duplicate combinations.

Example 1:
```c++
Input: candidates = [10,1,2,7,6,1,5], target = 8,
A solution set is:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]
```
Example 2:
```c++
Input: candidates = [2,5,2,1,2], target = 5,
A solution set is:
[
  [1,2,2],
  [5]
]
```

#### Solution: DFS
##### C++
```c++
class Solution {
public:
    vector<vector<int>> combinationSum2(vector<int> &nums, int target) {
        sort(nums.begin(), nums.end());
        dfs(nums, target, 0);
        return this->result;
    }

private:
    vector<vector<int>> result;
    vector<int> path;

    void dfs(vector<int> &nums, int gap, int cur) {
        if (!gap) {
            result.push_back(path);
            return;
        }
        for (int i = cur; i < nums.size(); ++i) {
            if (gap < nums[i]) break;
            path.emplace_back(nums[i]);
            // next step starts from i+1 to avoid using the same number again
            dfs(nums, gap - nums[i], i + 1);
            path.pop_back();

            // Skip duplicates
            while (i < nums.size() - 1 && nums[i] == nums[i + 1]) 
                ++i;
        }
    }
};
```

##### Python3
```python
def combinationSum2(candidates: 'List[int]', target: 'int') -> 'List[List[int]]':
    def dfs(gap, cur):
        if gap == 0:
            result.append(path[:])
            return
        prev = None
        for i in range(cur, len(candidates)):
            if gap < candidates[i]:
                break

            # Skip duplicates
            if prev == candidates[i]:
                continue
            prev = candidates[i]

            path.append(candidates[i])
            # next step starts from i+1 to avoid using the same number again
            dfs(gap - candidates[i], i + 1)
            path.pop()

    result = []
    path = []
    candidates.sort()
    dfs(target, 0)
    return result
```

###### [Back to Front](#table-of-contents)
---



### [Combination Sum III](https://leetcode.com/problems/combination-sum-iii/)

Find all possible combinations of `k` numbers that add up to a number `n`, given that only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.

Note:
- All numbers will be positive integers.
- The solution set must not contain duplicate combinations.

Example 1:
```c++
Input: k = 3, n = 7
Output: [[1,2,4]]
```

Example 2:
```c++
Input: k = 3, n = 9
Output: [[1,2,6], [1,3,5], [2,3,4]]
```

#### Solution: DFS
##### C++
```c++
class Solution {
public:
    vector<vector<int>> combinationSum3(int k, int n) {
        dfs(k, n, 1);
        return this->result;
    }

private:
    vector<vector<int>> result;
    vector<int> path;

    void dfs(int n_left, int gap, int cur) {
        if (!gap && !n_left) {
            result.push_back(path);
            return;
        }
        if (n_left <= 0) return;

        for (int i = cur; i <= 9; ++i) {
            if (gap < i) break;
            path.emplace_back(i);
            dfs(n_left - 1, gap - i, i + 1);
            path.pop_back();
        }
    }
};
```

##### Python3
```python
def combinationSum3(k: 'int', n: 'int') -> 'List[List[int]]':
    def dfs(left, gap, cur):
        if left == 0 and gap == 0:
            result.append(path[:])
            return
        if left == 0:
            return
        for i in range(cur, 10):
            if gap < i:
                break
            path.append(i)
            dfs(left - 1, gap - i, i + 1)
            path.pop()

    result, path = [], []
    dfs(k, n, 1)
    return result
```
###### [Back to Front](#table-of-contents)
---



# Design

### [LRU Cache](https://leetcode.com/problems/lru-cache/)

Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: `get` and `put`.

- `get(key)` - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
- `put(key, value)` - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.

#### Solution

![LRU1](https://www.geeksforgeeks.org/wp-content/uploads/LRU1.png)
<!-- ![LRU2](https://www.geeksforgeeks.org/wp-content/uploads/LRU2.png) -->

##### C++
```c++
class LRUCache {
public:
    LRUCache(int capacity) {
        this->capacity_ = capacity;
    }

    int get(int key) {
        if (map_.find(key) == map_.end())
            return -1;

        // Transfer the element pointed by map_[key] from queue_ into queue_,
        // inserting it at queue_.begin()
        queue_.splice(queue_.begin(), queue_, map_[key]);

        map_[key] = queue_.begin();
        return map_[key]->value;
    }

    void put(int key, int value) {
        if (map_.find(key) == map_.end()) { // if key is NOT in memory queue
            if (queue_.size() == capacity_) {
                map_.erase(queue_.back().key);
                queue_.pop_back();
            }
            queue_.push_front(CacheNode(key, value));
            map_[key] = queue_.begin();
        } else { // if key is in memory queue
            map_[key]->value = value;
            queue_.splice(queue_.begin(), queue_, map_[key]);
            map_[key] = queue_.begin();
        }
    }

private:
    struct CacheNode {
        int key, value;

        CacheNode(int k, int v) : key(k), value(v) {}
    };

    list<CacheNode> queue_;
    unordered_map<int, list<CacheNode>::iterator> map_;
    int capacity_;
};
```

##### Python3
```python
class LRUCache:
    def __init__(self, capacity: int):
        self._map = OrderedDict()
        self._capacity = capacity

    def get(self, key: int) -> int:
        if key not in self._map:
            return -1
        price = self._map.pop(key)
        self._map[key] = price
        return price

    def put(self, key: int, value: int) -> None:
        if key in self._map:
            self._map.pop(key)
        if len(self._map) == self._capacity:
            self._map.popitem(last=False)
        self._map[key] = value
```


###### [Back to Front](#table-of-contents)
---
