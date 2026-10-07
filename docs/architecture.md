# System Architecture

## 1. Project Overview

The Intelligent Disk Scheduling system is designed to improve disk I/O
performance by combining traditional disk scheduling algorithms,
predictive caching, and reinforcement learning.

The system will first establish traditional disk scheduling algorithms
as baseline approaches. Predictive caching and reinforcement learning
will then be introduced and evaluated against these baselines.

---

## 2. Basic Disk I/O Flow

A disk request follows this general flow:

Application
    ↓
I/O Request
    ↓
Request Queue
    ↓
Disk Scheduler
    ↓
Disk Head
    ↓
Disk

The application generates read/write requests. These requests are placed
in a request queue. The disk scheduler determines the order in which the
requests should be serviced.

---

## 3. Disk Request

Each disk request will contain information such as:

- Request ID
- Disk cylinder/position
- Arrival time
- Request type (read/write)

Example:

```text
Request ID: 5
Position: 122
Arrival Time: 4.2
Type: READ