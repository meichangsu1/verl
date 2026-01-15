# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Create a simple multi-turn dataset for testing
"""

import argparse
import copy
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/multiturn")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--num_conversations", type=int, default=100)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    args = parser.parse_args()

    # Create example conversations
    base_conversations = []

    # Conversation 1
    base_conversations.append(
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
                {"role": "user", "content": "And what about Germany?"},
                {"role": "assistant", "content": "The capital of Germany is Berlin."},
            ]
        }
    )

    # Conversation 2
    base_conversations.append(
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Can you explain quantum computing?"},
                {
                    "role": "assistant",
                    "content": "Quantum computing is a type of computing that uses quantum-mechanical phenomena, "
                    "such as superposition and entanglement, to perform operations on data.",
                },
                {"role": "user", "content": "How is it different from classical computing?"},
                {
                    "role": "assistant",
                    "content": "Classical computing uses bits that are either 0 or 1, while quantum computing uses "
                    "quantum bits or qubits that can exist in multiple states simultaneously due to superposition.",
                },
            ]
        }
    )

    # Conversation 3
    base_conversations.append(
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a simple Python function to calculate factorial."},
                {
                    "role": "assistant",
                    "content": (
                        "```python\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        "
                        "return n * factorial(n-1)\n```\n\nThis is a recursive function to calculate the "
                        "factorial of a number."
                    ),
                },
                {"role": "user", "content": "Can you make it iterative instead?"},
                {
                    "role": "assistant",
                    "content": (
                        "```python\ndef factorial(n):\n    result = 1\n    for i in range(1, n+1):\n        "
                        "result *= i\n    return result\n```\n\nThis is an iterative version of the factorial function."
                    ),
                },
            ]
        }
    )

    def clone_conversation(conversation, idx):
        cloned = copy.deepcopy(conversation)
        for msg in cloned["messages"]:
            if msg["role"] == "user":
                msg["content"] = f"{msg['content']} (example {idx})"
                break
        return cloned

    # Create a larger dataset by cycling base conversations.
    if args.num_conversations < 1:
        raise ValueError("num_conversations must be >= 1")

    conversations = [
        clone_conversation(base_conversations[i % len(base_conversations)], i)
        for i in range(args.num_conversations)
    ]

    # Create train and test datasets
    if len(conversations) == 1:
        train_data = conversations
        test_data = []
    else:
        train_size = int(len(conversations) * args.train_ratio)
        train_size = max(1, min(train_size, len(conversations) - 1))
        train_data = conversations[:train_size]
        test_data = conversations[train_size:]

    # Create output directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet files
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_df.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Handle HDFS if specified
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs

            makedirs(args.hdfs_dir)
            copy(src=local_dir, dst=args.hdfs_dir)
        except ImportError:
            print("Warning: HDFS support not available. Skipping HDFS copy.")

    # Print statistics
    print(f"Train dataset size: {len(train_df)}")
    print(f"Test dataset size: {len(test_df)}")
    print(f"Data saved to {local_dir}")


if __name__ == "__main__":
    main()
