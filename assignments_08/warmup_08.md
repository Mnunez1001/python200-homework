# Week 8 Warmup — Cloud Concepts

## Cloud Concepts Question 1

**What is the core economic model of cloud computing, and how does it differ from owning your own servers?**

Cloud computing lets you rent computing resources instead of buying and maintaining your own servers. Many services charge based on usage, while others use subscriptions or reserved capacity. Owning servers requires paying for hardware upfront and maintaining it even when it is not being fully used.

## Cloud Concepts Question 2

**What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.**

Vertical scaling means giving one machine more resources, such as a faster CPU or GPU and more RAM. For example, I might choose a more powerful machine when a model-training job needs more GPU memory.

Horizontal scaling means adding more machines and distributing work among them. For example, I might add servers to handle more visitors to a website, provided the application can distribute requests across those servers.

**For the three scenarios below, identify which type of scaling applies and why.**

- **A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.** Horizontal scaling applies because additional servers can share the increased traffic.
- **A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.** Vertical scaling applies because they are increasing the resources available on one machine.
- **A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.** Horizontal scaling applies because multiple machines can process different files in parallel.

## Cloud Concepts Question 3

**Classify each item as IaaS, PaaS, SaaS, or BaaS, and explain your reasoning.**

| Service | Classification | Reasoning |
|---      |---             |---        |
| Gmail   | SaaS           | It provides a complete email application that users access without managing its servers or software updates. |
| Azure Virtual Machines   | IaaS      | It provides virtual machines, while the customer manages the operating system and installed software. |
| AWS S3  | IaaS           | It provides object-storage infrastructure, while the customer manages stored data and access permissions. |
| GitHub Codespaces | PaaS | It provides a managed development environment where developers work on their code without maintaining the underlying host infrastructure. |
| Snowflake| SaaS          | It provides a managed data-analysis service without requiring customers to install or maintain the database software or servers. |
| Supabase | BaaS          | It provides backend features such as a database, authentication, storage, and APIs for applications. |

**Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example and describe what you, as the developer, are responsible for managing.**

### IaaS — Infrastructure as a Service

IaaS provides resources such as virtual machines, networking, and storage. With Azure Virtual Machines, I am responsible for the operating system, updates, installed software, application code, and appropriate security settings; the provider maintains the physical infrastructure.

### PaaS — Platform as a Service

PaaS provides a managed environment for developing or running software, reducing the infrastructure work I need to do. With GitHub Codespaces, I manage my code, project dependencies, and development-environment configuration, while GitHub manages the underlying hosting infrastructure.

### SaaS — Software as a Service

SaaS provides a complete application that I can use without installing or maintaining its underlying infrastructure. With Gmail, I manage my account, settings, and how I handle information, while Google maintains the application and servers.

## Cloud Concepts Question 4

**What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like AWS or GCP directly? What do you gain, and what do you give up?**

A managed data platform brings together tools for storing, processing, and analyzing data while handling much of the underlying infrastructure. AWS and GCP offer a broader collection of services, but building an equivalent system directly can require more configuration and integration work.

I gain convenience, integrated tools, and less maintenance. In exchange, I give up some control over the underlying system, may become dependent on platform-specific features, and may pay extra for that convenience. AWS and GCP also offer managed services, so the difference depends on which services I choose.

## Cloud Concepts Question 5

**The lesson describes a situation where the cloud is probably not the right choice. What is that situation?**

The cloud may be unnecessary when a small project's storage and computation fit comfortably on my computer and it does not need continuous availability or shared remote access. For example, running a small local prototype can be simpler and cheaper than setting up cloud services.

# Part 2: Warmup — Cloud Landscape

## Cloud Landscape Question 1

**Name the three hyperscalers. For each, write one sentence describing its primary strength and the type of organization most likely to use it.**

- **Amazon Web Services (AWS):** AWS offers a broad, mature collection of cloud services, making it a common choice for startups and large organizations that need many different infrastructure tools. It's the most popular one and it dominates the market.
- **Google Cloud Platform (GCP):** GCP is particularly strong in data analytics and machine learning, making it attractive to organizations building data-intensive applications and AI systems.
- **Microsoft Azure:** Azure integrates closely with Microsoft's business tools and identity systems, making it a natural choice for enterprises and government organizations already using Microsoft technologies.


## Cloud Landscape Question 2

**The lesson explains why this course switched from Microsoft Azure to Supabase. It gives three concrete reasons. Summarize each reason in your own words — one sentence each.**

1. **Access:** Students can create their own Supabase projects without waiting for an approval or provisioning process, making it easier to get started.
2. **Pedagogical fit:** Supabase uses a PostgreSQL relational database, allowing students to practice tables, SQL, and other database concepts that transfer to many systems.
3. **Pipeline coherence:** The course can connect weather records, predictions, and later pipeline steps through related tables in one database, reinforcing how the pieces work together.

**What does this suggest about how you should evaluate a cloud tool when starting a new project?**

I should choose a tool based on the project's needs rather than its popularity. For a beginner, clear documentation and straightforward setup can make learning easier, but I should also consider cost, reliability, security, and how well the tool connects with the rest of the project. If a simpler tool meets the requirements with fewer complications, it may be the better choice.

## Cloud Landscape Question 3

**For each scenario, identify the service category from the taxonomy table and name one specific provider or product that offers it.**

| Scenario | Service category | Example product |
|---       |               ---|              ---|
| You need to store 10 TB of image files and retrieve them by filename from any machine. | Object storage | Amazon S3 |
| You need to run an ML training job on a GPU for four hours, then shut it down. | ML platform | Amazon SageMaker AI |
| You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets. | Serverless compute | AWS Lambda, exposed through Amazon API Gateway |
| You need to send structured data to a large language model and get a text response back. | LLM API | Amazon Bedrock |

Object storage retrieves files by an object key, which can act like a filename; access still requires appropriate permissions. A GPU virtual machine could also handle the training scenario, but an ML platform adds tools specifically for managing training jobs.

## Cloud Landscape Question 4

**Describe a simple data project of your own design and sketch a plausible stack using services from at least two different providers or products from the taxonomy table.**

I would build a weather dashboard that collects daily observations, stores them, and generates a short explanation of whether the conditions look suitable for running.

| Project component | Service category | Provider or product |
|---|---|---|
| Run the Python job that fetches and cleans weather data | Serverless compute | AWS Lambda |
| Store the original weather responses | Object storage | Amazon S3 |
| Store cleaned observations and running predictions | Managed relational DB | Supabase PostgreSQL |
| Generate a plain-language explanation of the results | LLM API | Amazon Bedrock |

**Is there a benefit to consolidating to one provider, and what would you give up if you did?**

Using one provider could simplify billing, permissions, monitoring, and connections between services, and it might reduce cross-provider data-transfer costs. For example, I could replace Supabase with Amazon RDS to keep this stack within AWS. However, I would give up Supabase's particular developer experience and integrated backend features, and I could become more dependent on one provider's ecosystem.