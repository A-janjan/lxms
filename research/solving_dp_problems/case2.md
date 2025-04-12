Reason You are a algorithmic problem solver. you should give your answer in pythonic code. be precise and careful about your solution.


# 2008. Maximum Earnings From Taxi


There are n points on a road you are driving your taxi on. The n points on the road are labeled from 1 to n in the direction you are going, and you want to drive from point 1 to point n to make money by picking up passengers. You cannot change the direction of the taxi.

The passengers are represented by a 0-indexed 2D integer array rides, where rides[i] = [starti, endi, tipi] denotes the ith passenger requesting a ride from point starti to point endi who is willing to give a tipi dollar tip.

For each passenger i you pick up, you earn endi - starti + tipi dollars. You may only drive at most one passenger at a time.

Given n and rides, return the maximum number of dollars you can earn by picking up the passengers optimally.

Note: You may drop off a passenger and pick up a different passenger at the same point.

Example 1:

Input: n = 5, rides = [[2,5,4],[1,5,1]]
Output: 7
Explanation: We can pick up passenger 0 to earn 5 - 2 + 4 = 7 dollars.
Example 2:

Input: n = 20, rides = [[1,6,1],[3,10,2],[10,12,3],[11,12,2],[12,15,2],[13,18,1]]
Output: 20
Explanation: We will pick up the following passengers:
- Drive passenger 1 from point 3 to point 10 for a profit of 10 - 3 + 2 = 9 dollars.
- Drive passenger 2 from point 10 to point 12 for a profit of 12 - 10 + 3 = 5 dollars.
- Drive passenger 5 from point 13 to point 18 for a profit of 18 - 13 + 1 = 6 dollars.
We earn 9 + 5 + 6 = 20 dollars in total.

Constraints:

1 <= n <= 105
1 <= rides.length <= 3 * 104
rides[i].length == 3
1 <= starti < endi <= n
1 <= tipi <= 105

write your answer in this style:

class Solution(object):
    def maxTaxiEarnings(self, n, rides):
        """
        :type n: int
        :type rides: List[List[int]]
        :rtype: int
        """
        
---

## soltion

```python
class Solution(object):
    def maxTaxiEarnings(self, n, rides):
        """
        :type n: int
        :type rides: List[List[int]]
        :rtype: int
        """
        # We'll build a mapping from each start point to the rides (end, tip)
        from collections import defaultdict
        rides_by_start = defaultdict(list)
        for s, e, tip in rides:
            rides_by_start[s].append((e, tip))
            
        # dp[i] will represent the maximum earnings from point i to n.
        # We use a dp array of size n + 2 so that dp[n+1] is available (and defaults to 0).
        dp = [0] * (n + 2)
        
        # Process the points in reverse order.
        for i in range(n, 0, -1):
            # Option 1: Skip taking any ride starting at i; move to next point.
            dp[i] = dp[i + 1]
            
            # Option 2: Try each ride starting from point i.
            if i in rides_by_start:
                for end, tip in rides_by_start[i]:
                    # The earnings for this ride: (end - i + tip)
                    # plus the best earnings starting from the drop-off point 'end'.
                    dp[i] = max(dp[i], end - i + tip + dp[end])
                    
        return dp[1]
``` 

### Explanation

1. **Data Preparation:**  
   We construct a dictionary (`rides_by_start`) to quickly access all rides that start from a given point. This avoids scanning through all rides for each position on the road.

2. **Dynamic Programming Setup:**  
   We use a DP array `dp` where `dp[i]` holds the maximum earnings we can obtain starting from position `i` to `n`. We add one extra element (`dp[n+1]`) initialized to zero to handle the boundary when `i = n`.

3. **Reverse Iteration:**  
   We iterate from `n` down to `1`. At each point `i`, there are two choices:
   - **Skip the ride at `i`:** Then the best earnings would be `dp[i+1]`, the maximum earnings from the next point.
   - **Take any ride that starts at `i`:** For each ride `[i, end, tip]`, the earnings is computed as `(end - i + tip)`, and then we add the maximum earnings from point `end` which is `dp[end]`.

4. **Result:**  
   The answer is stored in `dp[1]`, which represents the maximum earnings starting from the first point.

This approach efficiently computes the answer in \(O(n + m)\) time where \(m\) is the number of rides, making it suitable given the problem constraints.

---

## result

![alt text](assets/image2.png)