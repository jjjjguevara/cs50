# Methods

## Literature Search Strategy

Our systematic review followed the PRISMA guidelines[^6], utilizing multiple databases:

* PubMed Central
* Web of Science
* PsycINFO
* Google Scholar

### Inclusion Criteria

1. Empirical studies published between 2010-2024
2. Peer-reviewed articles in English
3. Studies involving professional musicians
4. Clear reporting of neuroimaging methods

## Meta-Analysis Approach

Statistical analyses were conducted using R (version 4.2.1)[^7], with the following packages:

```r
library(metafor)
library(dplyr)
library(ggplot2)
```

Effect sizes were calculated using Hedges' g to correct for small sample bias[^8].

[^6]: Page, M. J., McKenzie, J. E., Bossuyt, P. M., Boutron, I., Hoffmann, T. C., Mulrow, C. D., ... & Moher, D. (2021). The PRISMA 2020 statement: An updated guideline for reporting systematic reviews. *BMJ*, 372.
[^7]: R Core Team (2021). R: A language and environment for statistical computing. R Foundation for Statistical Computing, Vienna, Austria.
[^8]: Borenstein, M., Hedges, L. V., Higgins, J. P., & Rothstein, H. R. (2021). Introduction to meta-analysis. John Wiley & Sons.
