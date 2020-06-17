from GraphEmbedding.ge.models import node2vec


class Node2Vec():
    def __init__(self, walk_length=15, num_walks=200, p=0.25, q=4, workers=1):
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers
        self.model = None

    def build_model(self, G):
        self.model = node2vec.Node2Vec(
            G, walk_length=self.walk_length, num_walks=self.num_walks,
            p=self.p, q=self.q, workers=self.workers
        )

    def train(self, embed_size=50, window_size=5, iter=20):
        assert self.model is not None
        self.model.train(embed_size, window_size, iter)
        return self.model.get_embeddings()


if __name__ == "__main__":
    from utils.data_loader import DataLoader
    test = DataLoader('E:/python-workspace/CityRoadPrediction/data_20200610/test/')
    model = Node2Vec()
    model.build_model(test.build_source_graph(0))
    embeds = model.train()
