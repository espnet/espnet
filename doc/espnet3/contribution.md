## ESPnet3 Developer Contribution Guide

This document explains how to contribute to the development of espnet3. It covers the following scenarios:

1. Requesting new features
2. Reporting and fixing bugs
3. Asking questions or other contribution types

---

### 1. Requesting New Features

Typical feature requests might include adding a new trainer. If you are adding a new model or task for ESPnet2, that should be proposed and discussed in the ESPnet2 repository instead.

To request a feature, please open either:
- An **Issue**
- A **Pull Request (PR)**
- A **Discussion**

Please include the following in your submission:

- **What**: Clearly describe what feature you are requesting. This helps others assess the feasibility of the proposal.
- **Why**: Explain the motivation behind the request. This helps the maintainers evaluate the importance and priority of the feature.
- **Test Cases**: Describe test cases that demonstrate how the feature will be used. Even if the developer ends up writing the test, understanding the user's intent helps shape meaningful tests.

If you open a Pull Request, please include:
- A summary of the feature and its purpose
- The test cases included
- Explanation of the testing approach

For Discussions, even if your idea is not yet finalized, clearly state why you're opening the discussion and what problems you are trying to solve.

---

### 2. Reporting and Fixing Bugs

If you encounter a bug, open either:
- An **Issue**
- A **Pull Request** (if you already have a fix)

For bug reports, include the following:

- **Description**: What exactly is the bug?
- **Environment**: Include Python version, PyTorch version, OS, GPU model, etc.
- **Minimal Reproducible Code**: A small example that reproduces the bug. Please verify that it occurs with the latest `main` branch.
- **Error Log**: Include the full stack trace, not just the last error line. This may help uncover related issues.
- **Test Case**: If you're submitting a fix, include a test that fails before the fix and passes after.

For Pull Requests:
- Make sure to include all of the above so that maintainers can understand the problem, verify the fix, and ensure it doesn't cause regressions elsewhere.

---

### 3. Other Requests or Questions

If you have general questions, documentation requests, or ideas that do not fall under bug fixes or features, open an **Issue** or **Discussion**.

Please provide:

- **What**: Clearly describe what you're asking or proposing.
- **Why**: Explain the context or motivation for your question/request. This helps assess whether documentation updates or feature expansions are needed.

---

Thank you for helping improve ESPnet3!

