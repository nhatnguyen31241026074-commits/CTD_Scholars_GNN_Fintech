import random
from datetime import datetime, timedelta
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_transactions(
	num_users: int,
	num_transactions: int,
	start_date: str,
	end_date: str,
) -> pd.DataFrame:
	"""Generate a synthetic transactions DataFrame.

	Parameters
	- num_users: number of distinct users to create (e.g. 100)
	- num_transactions: number of transactions to simulate (e.g. 1000)
	- start_date: simulation start date in YYYY-MM-DD
	- end_date: simulation end date in YYYY-MM-DD (inclusive)

	Returns
	- pd.DataFrame with columns ['sender_id','receiver_id','amount','timestamp','transaction_type']

	Behavior
	- user ids: 'user_1' .. 'user_{num_users}'
	- sender and receiver sampled uniformly without replacement per tx (sender != receiver)
	- amount: integer VND between 10_000 and 5_000_000 inclusive
	- timestamp: random datetime between start_date (00:00:00) and end_date (23:59:59)
	- transaction_type: one of ['P2P','Nạp tiền','Rút tiền'] with higher weight for 'P2P'
	"""

	if num_users < 2:
		raise ValueError("num_users must be at least 2 to create valid sender/receiver pairs")

	# create user ids
	users: List[str] = [f"user_{i+1}" for i in range(num_users)]

	# parse date strings
	try:
		start_dt = datetime.strptime(start_date, "%Y-%m-%d")
		end_dt = datetime.strptime(end_date, "%Y-%m-%d")
	except Exception as e:
		raise ValueError("start_date and end_date must be in YYYY-MM-DD format") from e

	if end_dt < start_dt:
		raise ValueError("end_date must be the same or after start_date")

	# make end inclusive until 23:59:59 of end_date
	end_dt_inclusive = end_dt + timedelta(days=1) - timedelta(seconds=1)
	total_seconds = int((end_dt_inclusive - start_dt).total_seconds())

	records = []
	tx_types = ["P2P", "Nạp tiền", "Rút tiền"]
	# give P2P a higher probability
	probs = [0.75, 0.125, 0.125]

	for _ in range(num_transactions):
		# choose distinct sender and receiver
		sender, receiver = random.sample(users, 2)

		# random amount (integer VND)
		amount = int(np.random.randint(10_000, 5_000_000 + 1))

		# random timestamp between start and end inclusive
		rand_sec = random.randint(0, total_seconds)
		timestamp = start_dt + timedelta(seconds=rand_sec)

		# transaction type with weighted probability
		transaction_type = np.random.choice(tx_types, p=probs)

		records.append(
			{
				"sender_id": sender,
				"receiver_id": receiver,
				"amount": amount,
				"timestamp": timestamp,
				"transaction_type": transaction_type,
			}
		)

	df = pd.DataFrame.from_records(records, columns=["sender_id", "receiver_id", "amount", "timestamp", "transaction_type"])

	# sort by timestamp for easier downstream use
	df = df.sort_values("timestamp").reset_index(drop=True)

	return df


__all__ = ["generate_synthetic_transactions"]


if __name__ == "__main__":
	# Quick runtime test: generate a small sample and save to ../data
	print("Bắt đầu tạo dữ liệu giao dịch mẫu...")
	df_transactions = generate_synthetic_transactions(
		num_users=100,
		num_transactions=1000,
		start_date="2025-01-01",
		end_date="2025-01-31",
	)

	# construct an output path relative to the repository root (one level up from src)
	script_dir = Path(__file__).resolve().parent
	output_path = script_dir.parent / "data" / "sample_transactions.csv"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	df_transactions.to_csv(output_path, index=False)
	print(f"Đã tạo và lưu {len(df_transactions)} giao dịch vào file: {output_path}")
	print("5 dòng dữ liệu đầu tiên:")
	print(df_transactions.head())

