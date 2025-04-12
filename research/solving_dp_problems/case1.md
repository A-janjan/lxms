# 2002. Maximum Product of the Length of Two Palindromic Subsequences

Given a string s, find two disjoint palindromic subsequences of s such that the product of their lengths is maximized. The two subsequences are disjoint if they do not both pick a character at the same index.

Return the maximum possible product of the lengths of the two palindromic subsequences.

A subsequence is a string that can be derived from another string by deleting some or no characters without changing the order of the remaining characters. A string is palindromic if it reads the same forward and backward.

Example 1:
Input: s = "leetcodecom"
Output: 9
Explanation: An optimal solution is to choose "ete" for the 1st subsequence and "cdc" for the 2nd subsequence.
The product of their lengths is: 3 * 3 = 9.

Example 2:

Input: s = "bb"
Output: 1
Explanation: An optimal solution is to choose "b" (the first character) for the 1st subsequence and "b" (the second character) for the 2nd subsequence.
The product of their lengths is: 1 * 1 = 1.


Example 3:

Input: s = "accbcaxxcxx"
Output: 25
Explanation: An optimal solution is to choose "accca" for the 1st subsequence and "xxcxx" for the 2nd subsequence.
The product of their lengths is: 5 * 5 = 25.


Constraints:

2 <= s.length <= 12
s consists of lowercase English letters only.


---------------------------------------
## solution


Below is a Python solution that generates all possible subsequences using bit masks. For each mask we check if the selected subsequence is a palindrome. Then, we store the mask and its length if it is palindromic. Finally, we iterate over every pair of valid palindromic subsequences, ensuring they are disjoint (i.e. bit masks do not overlap), and update the maximum product of lengths accordingly.

```python
def maxProduct(s: str) -> int:
    n = len(s)
    palindromes = []
    
    # Iterate over all possible subsequences
    for mask in range(1, 1 << n):
        subseq = []
        for i in range(n):
            if mask & (1 << i):
                subseq.append(s[i])
        sub_str = "".join(subseq)
        # Check if the subsequence is a palindrome
        if sub_str == sub_str[::-1]:
            palindromes.append((mask, len(sub_str)))
    
    max_prod = 0
    L = len(palindromes)
    
    # Compare every pair of palindromic subsequences
    for i in range(L):
        mask1, len1 = palindromes[i]
        for j in range(i+1, L):
            mask2, len2 = palindromes[j]
            # They are disjoint if they don't share any index
            if mask1 & mask2 == 0:
                max_prod = max(max_prod, len1 * len2)
    
    return max_prod

# Example test cases
if __name__ == "__main__":
    test_cases = [
        ("leetcodecom", 9),
        ("bb", 1),
        ("accbcaxxcxx", 25)
    ]
    
    for s, expected in test_cases:
        result = maxProduct(s)
        print(f"Input: {s}\nOutput: {result}\nExpected: {expected}\n{'-'*40}")
```

### Explanation

1. **Generate Subsequences:**  
   We loop over all integers from 1 to \(2^n - 1\) (each integer represents a subset of indices in `s`) and use bit manipulation to build each subsequence.

2. **Palindrome Check:**  
   For each subsequence, the code checks whether it is a palindrome by comparing it to its reverse. If it is, we store the mask together with its length.

3. **Disjoint Pairing:**  
   We then compare each pair of stored palindromic subsequences. If their bit masks do not have any common bits (ensuring the chosen indices are disjoint), we compute the product of their lengths and update the maximum product.

4. **Output:**  
   Finally, the maximum product is returned.

You can execute this code to verify that it meets the problem requirements.


---

write the answer for this problem in this way:

class Solution(object):
    def maxProduct(self, s):
        """
        :type s: str
        :rtype: int
        """
        
---

Sure! Here's the solution rewritten in the format you requested:

```python
class Solution(object):
    def maxProduct(self, s):
        """
        :type s: str
        :rtype: int
        """
        n = len(s)
        palindromes = []
        
        # Generate all subsequences using bitmask
        for mask in range(1, 1 << n):
            subseq = []
            for i in range(n):
                if mask & (1 << i):
                    subseq.append(s[i])
            if subseq == subseq[::-1]:  # Check if palindromic
                palindromes.append((mask, len(subseq)))
        
        max_prod = 0
        L = len(palindromes)
        
        # Compare all pairs of disjoint palindromic subsequences
        for i in range(L):
            mask1, len1 = palindromes[i]
            for j in range(i + 1, L):
                mask2, len2 = palindromes[j]
                if mask1 & mask2 == 0:  # disjoint
                    max_prod = max(max_prod, len1 * len2)
        
        return max_prod
```

You can test it like this:

```python
sol = Solution()
print(sol.maxProduct("leetcodecom"))     # Output: 9
print(sol.maxProduct("bb"))              # Output: 1
print(sol.maxProduct("accbcaxxcxx"))     # Output: 25
```


---

## result

Accepted

Runtime: 8 ms