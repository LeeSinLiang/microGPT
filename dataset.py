from torch.utils.data import Dataset

class CorpusDataset(Dataset):
	def __init__(self, text, max_length):
		self.text = text
		self.max_length = max_length

	def __len__(self):
		return len(self.text)

	def __getitem__(self, idx):
		x = self.text[idx:idx+self.max_length]
		y = self.text[idx+1:idx+self.max_length+1]
		return (x, y)
