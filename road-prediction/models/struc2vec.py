from GraphEmbedding.ge.models import struc2vec


class Struc2Vec():
    def __init__(self, walk_length=30, num_walks=200, workers=1):
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.model = None

    def build_model(self, G):
        self.model = struc2vec.Struc2Vec(
            G, walk_length=self.walk_length, num_walks=self.num_walks, workers=self.workers
        )

    def train(self, embed_size=50, window_size=5, iter=10):
        assert self.model is not None
        self.model.train(embed_size, window_size, iter)
        return self.model.get_embeddings()


if __name__ == "__main__":
    from utils.data_loader import DataLoader
    test = DataLoader('E:/python-workspace/CityRoadPrediction/data_20200610/test/')
    model = Struc2Vec()
    model.build_model(test.build_source_graph(0))
    embeds = model.train()
