## Week 8 Project — Supabase Setup

The project has been set. Both tables have been created and the .env file has both keys present. One issue that I ran across was that the naming of some tabs or tolls have change so it was a little confusing at the beginning to undestand if i was in the right place or if I was getting the right key.

## Part B: Cloud Cost Analysis

### What does each scenario cost, and did the numbers surprise me?

I used the AWS Pricing Calculator with US East (N. Virginia) for both scenarios.

| Scenario | Service and configuration | Estimated monthly cost |
|---|---|---:|
| A | EC2 t3.micro, on-demand, 160 hours per month | $1.66 |
| B | EC2 p3.2xlarge, on-demand, 730 hours per month | $2,233.80 |
| B | RDS for SQL Server, db.m5.large, including the configured storage and options | $755.66 |
| B | S3 Standard, 1 TB of storage | $23.55 |
| **Scenario B total** | | **$3,013.01** |

Both estimates have $0 in upfront costs. These totals reflect the
configured services and usage assumptions, rather than a guaranteed bill.

I was surprised by the difference between a small instance used part-time
and a GPU instance running continuously. The database selection also
mattered: I selected RDS for SQL Server, which produced a higher estimate
than the PostgreSQL configuration discussed initially. This showed me why
an estimate needs to document the database engine and other settings,
not just the instance size.

### What does the comparison tell me about when a GPU is worth using?

The GPU instance accounts for most of Scenario B's monthly cost. It could
be worth using for a workload that benefits substantially from GPU
acceleration, such as training a large neural network, but it would
usually be unnecessary for a small weather pipeline or lightweight
classifier. The comparison also includes different operating hours:
Scenario A runs for 160 hours, while Scenario B runs for 730 hours.
Choosing appropriate hardware and stopping compute when it is no longer
needed can both make a substantial difference.

### Video

**Video link:** https://youtu.be/zOyTtus-Daw   