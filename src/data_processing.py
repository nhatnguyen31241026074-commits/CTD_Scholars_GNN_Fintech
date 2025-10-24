import random
from datetime import datetime, timedelta
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd

# Optional imports for graph creation. Keep optional so module imports even if torch is not installed.
try:
	import torch
	from torch_geometric.data import Data
except Exception:
	torch = None
	Data = None


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


def _encode_transaction_types(df: pd.DataFrame) -> dict:
	"""Create a mapping for transaction types to integer indices.

	Returns a dict mapping type->index and ensures consistent ordering.
	"""
	types = sorted(df['transaction_type'].unique().tolist())
	mapping = {t: i for i, t in enumerate(types)}
	return mapping


def _normalize_amounts(amounts: np.ndarray) -> np.ndarray:
	"""Min-max normalize amounts to [0,1]."""
	min_a = amounts.min()
	max_a = amounts.max()
	denom = (max_a - min_a) if max_a > min_a else 1.0
	return (amounts - min_a) / denom


def _time_hour_sin_cos(timestamps: pd.Series) -> np.ndarray:
	"""Compute cyclical hour features (sin, cos) from datetime-like series."""
	# Ensure datetime
	hours = timestamps.dt.hour.astype(float)
	radians = 2 * np.pi * hours / 24.0
	sin = np.sin(radians)
	cos = np.cos(radians)
	return np.vstack([sin.values, cos.values]).T


def _transaction_type_onehot(types: pd.Series, mapping: dict) -> np.ndarray:
	num = len(types)
	k = len(mapping)
	onehot = np.zeros((num, k), dtype=float)
	for i, t in enumerate(types):
		idx = mapping.get(t, None)
		if idx is not None:
			onehot[i, idx] = 1.0
	return onehot


def create_graph_snapshot(transactions_df: pd.DataFrame, user_list: List[str]) -> 'Data':
	"""Create a torch_geometric Data object from transactions.

	Parameters
	- transactions_df: DataFrame with columns ['sender_id','receiver_id','amount','timestamp','transaction_type']
	- user_list: list of all user ids (defines node ordering)

	Returns
	- torch_geometric.data.Data with x, edge_index, edge_attr
	"""
	if torch is None or Data is None:
		raise ImportError("create_graph_snapshot requires 'torch' and 'torch_geometric' to be installed")

	# create mapping user_id -> index
	user_to_idx = {u: i for i, u in enumerate(user_list)}
	num_nodes = len(user_list)

	# Ensure timestamps are datetime
	df = transactions_df.copy()
	if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
		df['timestamp'] = pd.to_datetime(df['timestamp'])

	# Edge index (2, N)
	senders = df['sender_id'].map(user_to_idx).astype(int).values
	receivers = df['receiver_id'].map(user_to_idx).astype(int).values
	edge_index = torch.tensor([senders, receivers], dtype=torch.long)

	# Edge attributes
	amounts = df['amount'].to_numpy(dtype=float)
	amounts_norm = _normalize_amounts(amounts).reshape(-1, 1)

	time_feats = _time_hour_sin_cos(df['timestamp'])  # Nx2

	type_map = _encode_transaction_types(df)
	type_onehot = _transaction_type_onehot(df['transaction_type'], type_map)  # NxK

	edge_attr_np = np.hstack([amounts_norm, time_feats, type_onehot])
	edge_attr = torch.tensor(edge_attr_np, dtype=torch.float)

	# Node features placeholder: Px1 tensor of ones
	x = torch.ones((num_nodes, 1), dtype=torch.float)

	data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
	# attach ancillary metadata for convenience
	data.user_to_idx = user_to_idx
	data.type_mapping = type_map

	return data


__all__ = ["generate_synthetic_transactions", "create_graph_snapshot"]


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

	print("\nBắt đầu tạo snapshot đồ thị...")
	all_users = pd.concat([df_transactions['sender_id'], df_transactions['receiver_id']]).unique().tolist()
	try:
		graph_snapshot = create_graph_snapshot(df_transactions, all_users)
		print("Đã tạo snapshot đồ thị:")
		print(graph_snapshot)
		# Optionally save snapshot
		# torch.save(graph_snapshot, script_dir.parent / "data" / "sample_graph_snapshot.pt")
	except Exception as e:
		print("Không thể tạo snapshot đồ thị:", str(e))
		print("Để tạo snapshot bạn cần cài đặt 'torch' và 'torch-geometric'. Ví dụ: pip install torch torch-geometric (vui lòng tham khảo tài liệu chính thức để chọn phiên bản tương thích).")

