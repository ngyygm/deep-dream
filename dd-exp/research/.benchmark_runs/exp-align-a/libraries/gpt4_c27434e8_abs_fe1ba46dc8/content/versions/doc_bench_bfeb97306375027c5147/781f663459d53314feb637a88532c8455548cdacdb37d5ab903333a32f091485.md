[sharegpt_LFb4L6G_0:1] user: Web search results:

[1] "Spot VMs are ideal for the following types of workloads: Dev/test environments, including continuous integration and continuous delivery (CI/CD) workloads. Select high-performance computing scenarios, batch processing jobs, or visual rendering applications. Big data, analytics, container-based, and large-scale stateless applications."
URL: https://azure.microsoft.com/en-us/products/virtual-machines/spot/

[2] "An Azure Spot Virtual Machine offers no high availability guarantees. At any point in time when Azure needs the capacity back, the Azure infrastructure will evict Azure Spot Virtual Machines with 30-seconds notice. Eviction policy VMs can be evicted based on capacity or the max price you set."
URL: https://learn.microsoft.com/en-us/azure/virtual-machines/spot-vms

[3] "Spot VMs in a managed instance group This page describes Spot VMs: virtual machine (VM) instances that are excess Compute Engine capacity. Spot VMs have significant discounts, but Compute..."
URL: https://cloud.google.com/compute/docs/instances/spot
Current date: 1/29/2023

Instructions: Using the provided web search results, write a comprehensive reply to the given query. Make sure to cite results using [[number](URL)] notation after the reference. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject.
Query: what is a spot VM?
[sharegpt_LFb4L6G_0:2] assistant: A Spot VM is a type of virtual machine (VM) that is made up of excess computing capacity offered by cloud service providers such as Microsoft Azure and Google Cloud Compute Engine. It is designed to handle specific types of workloads such as dev/test environments, high-performance computing scenarios, batch processing jobs, big data, analytics, container-based and large-scale stateless applications [[1](https://azure.microsoft.com/en-us/products/virtual-machines/spot/)].

Note that the use of a Spot VM is subject to capacity availability and can be evicted at any time by the cloud provider with a 30-second notice when more capacity is needed [[2](https://learn.microsoft.com/en-us/azure/virtual-machines/spot-vms)]. Despite this limitation, using a Spot VM can provide significant discounts compared to other virtual machine instances [[3](https://cloud.google.com/compute/docs/instances/spot)].