# Attention Head Dimension Fix

## Problem
RuntimeError: shape '[188, 5, 8, 5]' is invalid for input of size 38540

The issue occurred because:
- Feature dimension was 41 (31 scaled + 10 encoded features)
- Attention heads was set to 8
- 41 / 8 = 5.125 (not an integer)
- The model tried to reshape to 8 heads × 5 dimensions = 40, missing 1 dimension

## Solution
Modified `TransEmbedding` in `rgtan_model.py` to handle non-divisible dimensions:

```python
# Before:
self.att_head_size = int(in_feats_dim / att_head_num)
self.total_head_size = in_feats_dim

# After:
self.att_head_size = in_feats_dim // att_head_num
self.total_head_size = self.att_head_size * att_head_num
```

This ensures:
- Head size is always an integer (floor division)
- Total head size might be slightly less than input dimension
- Linear layers are adjusted accordingly

## Alternative Solutions

1. **Use 1 attention head**: Set `nei_att_heads: creditcard: 1`
2. **Pad features**: Add dummy features to make dimension divisible
3. **Use different architecture**: Skip attention for neighborhood features

The implemented solution is most flexible as it works with any feature dimension and attention head combination.