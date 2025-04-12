You are a algorithmic problem solver. you should give your answer in pythonic code. be precise and careful about your solution.

# 2218. Maximum Value of K Coins From Piles

There are n piles of coins on a table. Each pile consists of a positive number of coins of assorted denominations.

In one move, you can choose any coin on top of any pile, remove it, and add it to your wallet.

Given a list piles, where piles[i] is a list of integers denoting the composition of the ith pile from top to bottom, and a positive integer k, return the maximum total value of coins you can have in your wallet if you choose exactly k coins optimally.

 

Example 1:

Input: piles = [[1,100,3],[7,8,9]], k = 2
Output: 101
Explanation:
The above diagram shows the different ways we can choose k coins.
The maximum total we can obtain is 101.
Example 2:

Input: piles = [[100],[100],[100],[100],[100],[100],[1,1,1,1,1,1,700]], k = 7
Output: 706
Explanation:
The maximum total can be obtained if we choose all coins from the last pile.
 

Constraints:

n == piles.length
1 <= n <= 1000
1 <= piles[i][j] <= 10^5
1 <= k <= sum(piles[i].length) <= 2000

write your answer in this way:

class Solution(object):
    def maxValueOfCoins(self, piles, k):
        """
        :type piles: List[List[int]]
        :type k: int
        :rtype: int
        """
        
---

## answer

```python
class Solution(object):
    def maxValueOfCoins(self, piles, k):
        """
        :type piles: List[List[int]]
        :type k: int
        :rtype: int
        """
        # dp[j] will hold the maximum value achievable with j coins.
        dp = [0] * (k + 1)
        
        # Process each pile one by one.
        for pile in piles:
            # Compute prefix sums for the current pile.
            # prefix[x] is the total value if we take x coins from this pile.
            prefix = [0]
            for coin in pile:
                prefix.append(prefix[-1] + coin)
            
            # Update dp in reverse to avoid using a coin more than once.
            # We iterate from k down to 0, ensuring that each pile's coins
            # are only used once per state update.
            for j in range(k, -1, -1):
                # Try taking x coins from the current pile, where x is at most
                # the number of coins in the current pile and also at most j.
                for x in range(1, min(len(prefix), j + 1)):
                    dp[j] = max(dp[j], dp[j - x] + prefix[x])
                    
        return dp[k]
```


---

## result

![alt text](assets/image10.png)


hard problem but solved!