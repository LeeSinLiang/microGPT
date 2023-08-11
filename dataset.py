from torch.utils.data import Dataset

class CorpusDataset(Dataset):
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		text = self.data[idx]
		return text