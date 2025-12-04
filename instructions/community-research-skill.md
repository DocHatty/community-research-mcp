# Community Research MCP - LLM Interaction Guide

> **Purpose:** This skill document teaches you how to leverage the Community Research MCP server to find real-world, battle-tested solutions from developer communities. Master this, and you transform from a model with static knowledge into one with access to what developers are *actually* discussing, debugging, and solving right now.

---

## Table of Contents

1. [Core Philosophy](#core-philosophy)
2. [When to Use This Tool](#when-to-use-this-tool)
3. [The community_search Tool](#the-community_search-tool)
4. [Query Crafting Masterclass](#query-crafting-masterclass)
5. [Source Intelligence](#source-intelligence)
6. [Result Interpretation](#result-interpretation)
7. [Advanced Patterns](#advanced-patterns)
8. [Interactive Workflows](#interactive-workflows)
9. [Failure Handling](#failure-handling)
10. [Pro Tips & Nuggets](#pro-tips--nuggets)

---

## Core Philosophy

### What This Tool IS
- A gateway to **street-smart solutions** from Stack Overflow, GitHub Issues, Reddit, Hacker News, and 6+ other developer communities
- Real developers sharing **what actually worked** (not what should work in theory)
- Community-validated answers with upvotes, accepted answers, and battle scars
- Current discussions about bleeding-edge tools and recent releases

### What This Tool IS NOT
- Official documentation (use Context7 for that)
- Curated tutorials (those often skip the painful edge cases)
- Your static training data (this is live community knowledge)

### The Mindset Shift
```
OLD: "Here's how the docs say to do it..."
NEW: "Here's what 500 developers on Stack Overflow discovered when they actually tried it..."
```

---

## When to Use This Tool

### PERFECT Use Cases

| Scenario | Why Community Research Shines |
|----------|------------------------------|
| **Debugging errors** | Real devs posted the same error + their fix |
| **"Best practice" questions** | See what patterns survived production |
| **Library comparison** | Community discussions reveal real trade-offs |
| **Migration paths** | Others documented their upgrade pain points |
| **Performance issues** | Benchmarks and real-world numbers |
| **Edge cases** | The stuff docs don't mention |
| **"Has anyone done X?"** | Yes, probably. Let's find them. |
| **Integration patterns** | How people combined Tool A with Tool B |
| **Version-specific bugs** | GitHub Issues track what broke in v2.3.1 |

### When to COMBINE with Other Sources

| Need | Use Community Research + |
|------|-------------------------|
| API reference | Context7 documentation |
| Full code examples | Read the actual file from search results |
| Latest releases | WebSearch for changelogs |
| Deep architectural decisions | Follow GitHub Discussions links |

### When to SKIP This Tool

- Simple syntax questions (your training data covers this)
- Asking "what is X?" for well-known concepts
- When the user explicitly wants official documentation
- Trivial one-liner solutions

---

## The community_search Tool

### Function Signature

```python
community_search(
    query: str,           # REQUIRED: Your search query
    language: str = "python",  # Programming language context
    max_results: int = 15      # Results per source (max 25)
)
```

### Response Structure

```json
{
  "query": "FastAPI background tasks Redis",
  "language": "python",
  "total_results": 47,
  "sources_searched": 8,
  "results": {
    "stackoverflow": [
      {
        "title": "How to run background tasks in FastAPI with Redis?",
        "url": "https://stackoverflow.com/questions/...",
        "score": 89,
        "snippet": "Use BackgroundTasks for lightweight work, but for Redis...",
        "votes": 142,
        "answers": 7,
        "accepted": true
      }
    ],
    "github": [...],
    "reddit": [...],
    "hackernews": [...],
    "discourse": [...],
    "lobsters": [...]
  },
  "audit": [
    {"source": "stackoverflow", "count": 10, "duration_ms": 234, "error": null},
    {"source": "github", "count": 8, "duration_ms": 456, "error": null}
  ]
}
```

---

## Query Crafting Masterclass

### The Art of the Query

Your query is everything. A good query finds gold; a bad query finds noise.

### Query Patterns That Work

#### Pattern 1: Error-Driven Search
When the user has an error, include the exact error message:
```
# GOOD
"TypeError: Cannot read property 'map' of undefined React useEffect"

# BAD  
"React error with map"
```

#### Pattern 2: Technology Stack Search
Combine the technologies:
```
# GOOD
"FastAPI SQLAlchemy async session management"

# BAD
"database sessions"
```

#### Pattern 3: Comparison Search
When comparing options:
```
# GOOD
"Redis vs RabbitMQ Python background tasks comparison"

# BAD
"best message queue"
```

#### Pattern 4: Version-Specific Search
Include version numbers for recent issues:
```
# GOOD
"Next.js 14 app router middleware not working"

# BAD
"Next.js middleware issues"
```

#### Pattern 5: Behavioral Search
Describe the behavior, not just the technology:
```
# GOOD
"React component re-renders on every state change performance"

# BAD
"React performance"
```

### Query Enrichment Strategy

Before calling `community_search`, enrich the query with context:

```
User says: "My API is slow"

Your enriched query: "FastAPI slow response time performance optimization async"
                      ^^^^^^^                                         ^^^^^
                      (from chat context)                    (add technical terms)
```

### The Language Parameter

Always set the `language` parameter appropriately:

| Language | Effect |
|----------|--------|
| `python` | Prioritizes Python-specific Stack Overflow tags, PyPI discussions |
| `javascript` | Includes npm, Node.js, browser contexts |
| `typescript` | Adds TypeScript-specific results |
| `rust` | Includes Rust Discourse, crates.io discussions |
| `go` | Adds Go forums, golang subreddit priority |

---

## Source Intelligence

### Understanding Each Source's Strengths

#### Stack Overflow (Authority: 100)
- **Best for:** Definitive answers, accepted solutions, code snippets
- **Signal:** High vote count + accepted answer = trusted solution
- **Caveat:** Answers may be outdated; check dates

#### GitHub Issues (Authority: 90)
- **Best for:** Bug reports, workarounds, version-specific fixes
- **Signal:** Closed issues with comments = problem solved
- **Caveat:** May reference internal code you don't have

#### Hacker News (Authority: 85)
- **Best for:** Architecture discussions, tool comparisons, industry opinions
- **Signal:** High points + many comments = engaging discussion
- **Caveat:** Opinions may be strong; filter for technical substance

#### Reddit (Authority: 75)
- **Best for:** Quick tips, community sentiment, "what worked for me"
- **Signal:** Upvotes + detailed comments
- **Caveat:** Quality varies; prioritize r/programming over meme subs

#### Discourse Forums (Authority: 88)
- **Best for:** Language-specific deep dives (Rust, Elixir, Swift)
- **Signal:** Official community discussions, core team responses
- **Caveat:** Smaller result set; more niche

#### Lobsters (Authority: 83)
- **Best for:** High-signal technical discussions, curated community
- **Signal:** Invite-only community means higher average quality
- **Caveat:** Smaller community; fewer results

### Source Priority by Query Type

| Query Type | Prioritize Sources |
|------------|-------------------|
| Bug/Error fix | Stack Overflow > GitHub Issues > Reddit |
| Performance | GitHub Issues > Hacker News > Stack Overflow |
| Architecture | Hacker News > Lobsters > Discourse |
| Tool comparison | Reddit > Hacker News > Stack Overflow |
| Language-specific | Discourse > Stack Overflow > GitHub |

---

## Result Interpretation

### Quality Score Breakdown (0-100)

Results are scored on multiple factors:

```
Quality Score = 
    Source Authority (0-100 base)
  + Community Validation (votes, answers)
  + Recency Bonus (recent = better)
  + Specificity (code blocks, detail)
  + Evidence (links, benchmarks, metrics)
  - Staleness Penalty (old content)
```

### How to Present Results to Users

#### Tier 1: High Confidence (Score 80+)
```
"I found a highly-rated Stack Overflow answer with 142 upvotes that directly 
addresses your issue. The accepted solution suggests..."
```

#### Tier 2: Good Leads (Score 60-79)
```
"Several GitHub issues discuss this problem. The most relevant one shows a 
workaround that was merged in PR #1234..."
```

#### Tier 3: Worth Exploring (Score 40-59)
```
"There's an interesting discussion on Hacker News about this trade-off. 
The community is split, but here are the main arguments..."
```

#### Tier 4: Low Signal (Score <40)
```
"I found some mentions of this, but nothing definitive. You might want to 
check the official docs or try a more specific search."
```

### Synthesizing Multiple Sources

When you get results from multiple sources, synthesize them:

```
"Looking across Stack Overflow, GitHub, and Reddit:

1. **Consensus:** All sources agree that using async/await here is the right 
   approach (SO: 3 answers, GH: 2 issues, Reddit: 5 comments)

2. **Divergence:** There's debate about whether to use library X or Y:
   - Stack Overflow leans toward X (more upvotes)
   - Reddit community prefers Y (more recent discussions)

3. **Hidden Gem:** One GitHub issue mentions a config flag that isn't in 
   the docs yet: `ENABLE_TURBO_MODE=true`
"
```

---

## Advanced Patterns

### Pattern 1: Progressive Search Refinement

When first search is too broad:

```
Search 1: "React state management" (too generic)
         --> 200 results, mostly basic tutorials

Search 2: "React state management large application performance" (refined)
         --> 50 results, more targeted

Search 3: "React context vs Redux performance re-renders large app" (specific)
         --> 15 high-quality discussions
```

### Pattern 2: Error Message Dissection

For complex errors, break them down:

```
Error: "OperationalError: (psycopg2.OperationalError) FATAL: too many connections"

Search 1: "psycopg2 OperationalError too many connections"
Search 2: "PostgreSQL connection pool exhaustion Python"
Search 3: "SQLAlchemy connection pool size configuration"
```

### Pattern 3: Temporal Context

For version-sensitive queries:

```
User: "Next.js 14 app router issues"

Search: "Next.js 14 app router" (captures recent discussions)

Then filter results mentally:
- Results from 2024 = highly relevant
- Results from 2023 = may predate app router
- Results from 2022 = almost certainly outdated
```

### Pattern 4: Cross-Pollination

Combine community insights with other tools:

```
1. community_search("FastAPI middleware order")
   --> Finds discussion about middleware execution order

2. context7_get_docs("fastapi", "middleware")
   --> Gets official documentation

3. Synthesize: "The community found that middleware runs in reverse order 
   from how it's declared. The docs confirm this at [link], and here's 
   a workaround pattern from Stack Overflow..."
```

### Pattern 5: Debugging Session

For persistent bugs:

```
Step 1: Search the exact error
Step 2: Search the technology + "issue" or "bug"
Step 3: Search the technology + "workaround"
Step 4: Search the technology + version number
Step 5: If GitHub issue found, suggest user subscribe to it
```

---

## Interactive Workflows

### Workflow 1: The Debugging Assistant

```
User: "I'm getting a CORS error in my React app calling FastAPI"

You:
1. community_search("FastAPI CORS error React frontend", language="python")
2. Present top Stack Overflow solutions
3. Identify if it's a common misconfiguration
4. Provide the fix with community validation

Response: "This is one of the top 10 most asked FastAPI questions. 
From 47 Stack Overflow answers with 2000+ combined votes, here's 
what works: [solution]. The key insight from GitHub issues is that 
you also need to allow credentials if using cookies..."
```

### Workflow 2: Technology Decision Helper

```
User: "Should I use Celery or RQ for background tasks?"

You:
1. community_search("Celery vs RQ Python background tasks comparison")
2. community_search("Celery production issues", language="python")
3. community_search("RQ production issues", language="python")
4. Synthesize community sentiment

Response: "Based on 50+ discussions across Reddit and HN:
- Celery: More features, but 'operationally complex' (common complaint)
- RQ: 'Simpler but limited' - works great up to ~10k jobs/day
Community consensus: Start with RQ, migrate to Celery when you hit limits."
```

### Workflow 3: The Migration Guide

```
User: "I need to migrate from Express to Fastify"

You:
1. community_search("Express to Fastify migration guide")
2. community_search("Fastify gotchas coming from Express")
3. community_search("Fastify Express differences")

Response: "Here's what developers who've done this migration warn about:
1. Plugin system is different (GH issue #1234)
2. Middleware order matters more (Reddit thread)
3. Schema validation is stricter (Stack Overflow)

Migration checklist from community experience:
- [ ] Plugins before routes (HN discussion)
- [ ] Update all middleware signatures
- [ ] Test validation schemas thoroughly
"
```

### Workflow 4: The Performance Detective

```
User: "My Python script is slow when processing large files"

You:
1. community_search("Python large file processing performance")
2. community_search("Python memory efficient file reading")
3. community_search("Python generator vs list large data")

Response: "Community optimization patterns (ordered by impact):
1. Use generators instead of loading entire file (500+ upvotes on SO)
2. Process in chunks with pandas chunksize (GitHub gist)
3. Consider memory-mapped files for random access (HN discussion)
4. Profile first - often it's I/O not CPU (Reddit wisdom)
"
```

---

## Failure Handling

### When Sources Return No Results

```
Audit shows: {"stackoverflow": 0, "github": 0, ...}

Actions:
1. Reformulate query with synonyms
2. Remove language-specific terms
3. Try broader category terms
4. Tell user: "This seems to be a niche topic. Let me try..."
```

### When Sources Are Down

```
Audit shows: {"stackoverflow": 0, "error": "timeout"}

Actions:
1. Check if multiple sources failed (might be network issue)
2. Present results from sources that worked
3. Note: "Stack Overflow was unreachable; results may be incomplete"
```

### When Results Are All Outdated

```
All results are from 2019 or earlier

Actions:
1. Explicitly note the age of solutions
2. Search for "[technology] 2024" or "latest"
3. Warn user: "These solutions may not apply to current versions"
```

### When Results Conflict

```
Stack Overflow says X, GitHub says Y

Actions:
1. Present both viewpoints
2. Note community consensus if one is more upvoted
3. Suggest: "Try X first (more upvotes), fall back to Y if needed"
```

---

## Pro Tips & Nuggets

### Nugget 1: The "Accepted Answer Isn't Always Best" Rule
Stack Overflow's accepted answer might be outdated. Check if a newer answer has more upvotes.

### Nugget 2: GitHub Issue Comments Are Gold
The issue description is just the start. The real solution is often in comment #7 or #12.

### Nugget 3: Reddit Skepticism
Reddit loves to say "just use [simpler tool]". Take with a grain of salt for complex use cases.

### Nugget 4: HN Contrarianism
Hacker News loves to debate. Extract the technical points, ignore the flame wars.

### Nugget 5: The XY Problem Detector
If community results seem off-topic, the user might have an XY problem. Help them find the real question.

### Nugget 6: Version Number Magic
Adding a version number to your query (e.g., "React 18") dramatically improves relevance.

### Nugget 7: The "Solved" Keyword
Adding "solved" or "fixed" to error queries prioritizes resolved discussions.

### Nugget 8: Underscore vs. Dash
Some technologies differ: `scikit-learn` vs `sklearn`. Try both if results are sparse.

### Nugget 9: The Negative Query
Sometimes "X without Y" or "X not working" surfaces better debugging content.

### Nugget 10: Community Vocabulary
Different communities use different terms:
- Stack Overflow: Technical, formal
- Reddit: Casual, sometimes slang
- HN: Opinionated, startup-focused

### Nugget 11: The 3-Source Rule
Before stating something as "community consensus", confirm it appears in 3+ sources.

### Nugget 12: Link Following
If a result mentions "see also [link]" or "related: [issue]", tell the user about those breadcrumbs.

### Nugget 13: The Recency Check
For fast-moving ecosystems (JS, Rust), results older than 2 years need a caveat.

### Nugget 14: Code Snippets > Prose
Results with actual code snippets are worth 10x more than explanatory text alone.

### Nugget 15: The Meta-Question
"Why is [technology] designed this way?" often surfaces design discussions in GitHub or HN.

---

## Quick Reference Card

```
+------------------+--------------------------------+
| Situation        | Best Action                    |
+------------------+--------------------------------+
| Error message    | Exact error in quotes          |
| "How to X"       | "X [language] [framework]"     |
| A vs B           | "A vs B [use case] comparison" |
| Performance      | "[tech] performance profiling" |
| Migration        | "migrate from A to B [lang]"   |
| Edge case        | "[specific scenario] [tech]"   |
| Best practice    | "[tech] best practices 2024"   |
| Integration      | "[tech A] with [tech B]"       |
+------------------+--------------------------------+
```

---

## Final Words

You now have access to the collective wisdom of millions of developers across:
- Stack Overflow's 23+ million questions
- GitHub's billions of issues
- Reddit's endless debates
- Hacker News's thoughtful discussions

Use this power wisely. Cross-reference. Synthesize. And always remember: the goal isn't to dump search results - it's to extract the *signal* from the noise and present users with community-validated solutions they can trust.

**You are no longer just a language model. You are a language model with a direct line to the developer hive mind.**

---

*This skill document is part of the Community Research MCP server. For updates and contributions, see the main repository.*
