# Community Research MCP - Quick Reference

## Tool Call

```python
community_search(query, language="python", max_results=15)
```

## Query Templates

| Use Case | Template |
|----------|----------|
| Error | `"[exact error message] [technology]"` |
| How-to | `"how to [action] [technology] [context]"` |
| Compare | `"[A] vs [B] [use case] comparison"` |
| Debug | `"[technology] [symptom] not working"` |
| Migrate | `"migrate [from] to [to] [language]"` |
| Optimize | `"[technology] performance optimization"` |
| Best Practice | `"[technology] best practices [year]"` |

## Source Priority by Goal

```
Bug Fix:        SO > GitHub > Reddit
Performance:    GitHub > HN > SO  
Architecture:   HN > Lobsters > Discourse
Comparison:     Reddit > HN > SO
Language-deep:  Discourse > SO > GitHub
```

## Quality Tiers

| Score | Confidence | Action |
|-------|------------|--------|
| 80+   | High       | Present as trusted solution |
| 60-79 | Good       | Present as strong lead |
| 40-59 | Medium     | Present with caveats |
| <40   | Low        | Suggest alternative search |

## Quick Fixes

| Problem | Solution |
|---------|----------|
| No results | Broaden query, remove specifics |
| Outdated results | Add year, "latest", version |
| Too many results | Add technology stack details |
| Conflicting results | Present both, note vote counts |

## Power Keywords

- `"solved"` - Finds resolved issues
- `"fixed"` - Finds bug fixes  
- `"workaround"` - Finds temp solutions
- `"production"` - Filters for real-world usage
- `"performance"` - Surfaces benchmarks
- `"2024"` / `"2025"` - Filters recent content

## Red Flags to Note

- Stack Overflow accepted answer from 5+ years ago
- GitHub issue still open with no comments
- Reddit answer with 0 upvotes
- Solution requires deprecated library

## Synthesis Formula

```
1. Find common theme across 3+ sources
2. Note highest-voted solution
3. Identify version/date constraints
4. Present with confidence level
5. Include backup option if exists
```
