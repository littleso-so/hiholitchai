import datetime
import pandas as pd
import re
import os
import json
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict
config = {
    # item
    'num_rate': 6,
    'num_genre': 25,
    'num_director': 2186,
    'num_actor': 8030,
    'embedding_dim': 32,
    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,
    # user
    'num_gender': 2,
    'num_age': 7,
    'num_occupation': 21,
    'num_zipcode': 3402,
    # cuda setting
    'use_cuda': False,
    # model setting
    'inner': 1,
    'lr': 5e-5,
    'local_lr': 5e-6,
    'batch_size': 32,
    'num_epoch': 20,
    # candidate selection
    'num_candidate': 20,
}
states = ["warm_state", "user_cold_state", "item_cold_state", "user_and_item_cold_state"]

class movielens_1m(object):
    def __init__(self):
        self.user_data, self.item_data, self.score_data = self.load()

    def load(self):
        path = "E:/研三/研三上/代码/MeLU/MeLU-master/movielens/ml-1m"
        profile_data_path = "{}/users.dat".format(path)
        score_data_path = "{}/ratings.dat".format(path)
        item_data_path = "{}/movies_extrainfos.dat".format(path)
        #用户属性信息
        profile_data = pd.read_csv(
            profile_data_path, names=['user_id', 'gender', 'age', 'occupation_code', 'zip'], 
            sep="::", engine='python'
        )
        #物品属性信息
        item_data = pd.read_csv(
            item_data_path, names=['movie_id', 'title', 'year', 'rate', 'released', 'genre', 'director', 'writer', 'actors', 'plot', 'poster'], 
            sep="::", engine='python', encoding="utf-8"
        )
        #评分信息
        score_data = pd.read_csv(
            score_data_path, names=['user_id', 'movie_id', 'rating', 'timestamp'],
            sep="::", engine='python'
        )

        score_data['time'] = score_data["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x))
        score_data = score_data.drop(["timestamp"], axis=1)
        return profile_data, item_data, score_data
def item_converting(row, rate_list, genre_list, director_list, actor_list):
    rate_idx = torch.tensor([[rate_list.index(str(row['rate']))]]).long()
    genre_idx = torch.zeros(1, 25).long()
    for genre in str(row['genre']).split(", "):
        idx = genre_list.index(genre)
        genre_idx[0, idx] = 1
    director_idx = torch.zeros(1, 2186).long()
    for director in str(row['director']).split(", "):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx[0, idx] = 1
    actor_idx = torch.zeros(1, 8030).long()
    for actor in str(row['actors']).split(", "):
        idx = actor_list.index(actor)
        actor_idx[0, idx] = 1
    return torch.cat((rate_idx, genre_idx, director_idx, actor_idx), 1)


def user_converting(row, gender_list, age_list, occupation_list, zipcode_list): 
    gender_idx = torch.tensor([[gender_list.index(str(row['gender']))]]).long()
    age_idx = torch.tensor([[age_list.index(str(row['age']))]]).long()
    occupation_idx = torch.tensor([[occupation_list.index(str(row['occupation_code']))]]).long()
    zip_idx = torch.tensor([[zipcode_list.index(str(row['zip'])[:5])]]).long()
    return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1)


def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_

#master_path = "E:/研三/研三上/代码/MeLU/MeLU-master/movielens/ml"
def generate(master_path):
    #dataset_path = "E:/研三/研三上/代码/MeLU/MeLU-master/movielens/ml-1m"
    dataset_path = 'E:/Code/melunew/movielens/ml-1m'
    rate_list = load_list("{}/m_rate.txt".format(dataset_path))
    #rate_list
    genre_list = load_list("{}/m_genre.txt".format(dataset_path))
    #genre_list
    actor_list = load_list("{}/m_actor.txt".format(dataset_path))
    #actor_list
    director_list = load_list("{}/m_director.txt".format(dataset_path))
    #director_list
    gender_list = load_list("{}/m_gender.txt".format(dataset_path))
    #gender_list
    age_list = load_list("{}/m_age.txt".format(dataset_path))
    #age_list
    occupation_list = load_list("{}/m_occupation.txt".format(dataset_path))
    #occupation_list
    zipcode_list = load_list("{}/m_zipcode.txt".format(dataset_path))
    #zipcode_list
    if not os.path.exists("{}/warm_state/".format(master_path)):
        for state in states:
            os.mkdir("{}/{}/".format(master_path, state))
    if not os.path.exists("{}/log/".format(master_path)):
        os.mkdir("{}/log/".format(master_path))

    dataset = movielens_1m()

    # hashmap for item information
    if not os.path.exists("{}/m_movie_dict.pkl".format(master_path)):
        movie_dict = {}
        for idx, row in dataset.item_data.iterrows():
            m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)
            movie_dict[row['movie_id']] = m_info
        pickle.dump(movie_dict, open("{}/m_movie_dict.pkl".format(master_path), "wb"))
    else:
        movie_dict = pickle.load(open("{}/m_movie_dict.pkl".format(master_path), "rb"))
    # hashmap for user profile
    if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
        user_dict = {}
        for idx, row in dataset.user_data.iterrows():
            u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
            user_dict[row['user_id']] = u_info
        pickle.dump(user_dict, open("{}/m_user_dict.pkl".format(master_path), "wb"))
    else:
        user_dict = pickle.load(open("{}/m_user_dict.pkl".format(master_path), "rb"))

    for state in states:
        idx = 0
        if not os.path.exists("{}/{}/{}".format(master_path, "log", state)):
            os.mkdir("{}/{}/{}".format(master_path, "log", state))
        with open("{}/{}.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset_y = json.loads(f.read())
        for _, user_id in tqdm(enumerate(dataset.keys())):
            u_id = int(user_id)
            seen_movie_len = len(dataset[str(u_id)])
            indices = list(range(seen_movie_len))

            if seen_movie_len < 13 or seen_movie_len > 100:
                continue

            random.shuffle(indices)
            tmp_x = np.array(dataset[str(u_id)])
            tmp_y = np.array(dataset_y[str(u_id)])

            support_x_app = None
            for m_id in tmp_x[indices[:-10]]:
                m_id = int(m_id)
                tmp_x_converted = torch.cat((movie_dict[m_id], user_dict[u_id]), 1)
                try:
                    support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
                except:
                    support_x_app = tmp_x_converted

            query_x_app = None
            for m_id in tmp_x[indices[-10:]]:
                m_id = int(m_id)
                u_id = int(user_id)
                tmp_x_converted = torch.cat((movie_dict[m_id], user_dict[u_id]), 1)
                try:
                    query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
                except:
                    query_x_app = tmp_x_converted
            support_y_app = torch.FloatTensor(tmp_y[indices[:-10]])
            query_y_app = torch.FloatTensor(tmp_y[indices[-10:]])

            pickle.dump(support_x_app, open("{}/{}/supp_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(support_y_app, open("{}/{}/supp_y_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(master_path, state, idx), "wb"))
            with open("{}/log/{}/supp_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[:-10]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            with open("{}/log/{}/query_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[-10:]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            idx += 1

class item(torch.nn.Module):
    def __init__(self, config):
        super(item, self).__init__()
        self.num_rate = config['num_rate']
        self.num_genre = config['num_genre']
        self.num_director = config['num_director']
        self.num_actor = config['num_actor']
        self.embedding_dim = config['embedding_dim']

        self.embedding_rate = torch.nn.Embedding(
            num_embeddings=self.num_rate, 
            embedding_dim=self.embedding_dim
        )
        
        self.embedding_genre = torch.nn.Linear(
            in_features=self.num_genre,
            out_features=self.embedding_dim,
            bias=False
        )
        
        self.embedding_director = torch.nn.Linear(
            in_features=self.num_director,
            out_features=self.embedding_dim,
            bias=False
        )
        
        self.embedding_actor = torch.nn.Linear(
            in_features=self.num_actor,
            out_features=self.embedding_dim,
            bias=False
        )

    def forward(self, rate_idx, genre_idx, director_idx, actors_idx, vars=None):
        rate_emb = self.embedding_rate(rate_idx)
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)
        director_emb = self.embedding_director(director_idx.float()) / torch.sum(director_idx.float(), 1).view(-1, 1)
        actors_emb = self.embedding_actor(actors_idx.float()) / torch.sum(actors_idx.float(), 1).view(-1, 1)
        return torch.cat((rate_emb, genre_emb, director_emb, actors_emb), 1)


class user(torch.nn.Module):
    def __init__(self, config):
        super(user, self).__init__()
        self.num_gender = config['num_gender']
        self.num_age = config['num_age']
        self.num_occupation = config['num_occupation']
        self.num_zipcode = config['num_zipcode']
        self.embedding_dim = config['embedding_dim']

        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        )

        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )

        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim
        )

        self.embedding_area = torch.nn.Embedding(
            num_embeddings=self.num_zipcode,
            embedding_dim=self.embedding_dim
        )

    def forward(self, gender_idx, age_idx, occupation_idx, area_idx):
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)
        return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)
class user_preference_estimator(torch.nn.Module):
    def __init__(self, config):
        super(user_preference_estimator, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.fc1_in_dim = config['embedding_dim'] * 8
        self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']
        self.use_cuda = config['use_cuda']

        self.item_emb = item(config)
        self.user_emb = user(config)
        self.fc1 = torch.nn.Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = torch.nn.Linear(self.fc2_in_dim, self.fc2_out_dim)
        self.linear_out = torch.nn.Linear(self.fc2_out_dim, 1)

    def forward(self, x, training = True):
        rate_idx = Variable(x[:, 0], requires_grad=False)
        genre_idx = Variable(x[:, 1:26], requires_grad=False)
        director_idx = Variable(x[:, 26:2212], requires_grad=False)
        actor_idx = Variable(x[:, 2212:10242], requires_grad=False)
        gender_idx = Variable(x[:, 10242], requires_grad=False)
        age_idx = Variable(x[:, 10243], requires_grad=False)
        occupation_idx = Variable(x[:, 10244], requires_grad=False)
        area_idx = Variable(x[:, 10245], requires_grad=False)

        item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
        user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx)
        x = torch.cat((item_emb, user_emb), 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.linear_out(x)


class MeLU(torch.nn.Module):
    def __init__(self, config):
        super(MeLU, self).__init__()
        self.use_cuda = config['use_cuda']
        self.model = user_preference_estimator(config)
        self.local_lr = config['local_lr']
        self.store_parameters()
        self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.local_update_target_weight_name = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'linear_out.weight', 'linear_out.bias']

    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)
        self.fast_weights = OrderedDict()

    def forward(self, support_set_x, support_set_y, query_set_x, num_local_update):
        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.model(support_set_x)
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
            self.model.zero_grad()
            grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            # local update
            for i in range(self.weight_len):
                if self.weight_name[i] in self.local_update_target_weight_name:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
                else:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
        self.model.load_state_dict(self.fast_weights)
        query_set_y_pred = self.model(query_set_x)
        self.model.load_state_dict(self.keep_weight)
        return query_set_y_pred

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys, num_local_update):
        batch_sz = len(support_set_xs)
        losses_q = []
        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()
        for i in range(batch_sz):
            query_set_y_pred = self.forward(support_set_xs[i], support_set_ys[i], query_set_xs[i], num_local_update)
            loss_q = F.mse_loss(query_set_y_pred, query_set_ys[i].view(-1, 1))
            losses_q.append(loss_q)
        losses_q = torch.stack(losses_q).mean(0)
        self.meta_optim.zero_grad()
        losses_q.backward()
        self.meta_optim.step()
        self.store_parameters()
        return

    def get_weight_avg_norm(self, support_set_x, support_set_y, num_local_update):
        tmp = 0.
        if self.cuda():
            support_set_x = support_set_x.cuda()
            support_set_y = support_set_y.cuda()
        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.model(support_set_x)
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
            # unit loss
            loss /= torch.norm(loss).tolist()
            self.model.zero_grad()
            grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            for i in range(self.weight_len):
                # For averaging Forbenius norm.
                tmp += torch.norm(grad[i])
                if self.weight_name[i] in self.local_update_target_weight_name:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
                else:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
        return tmp / num_local_update

def training(melu, total_dataset, batch_size, num_epoch, model_save=True, model_filename=None):
    if config['use_cuda']:
        melu.cuda()

    training_set_size = len(total_dataset)
    melu.train()
    for _ in range(num_epoch):
        random.shuffle(total_dataset)
        num_batch = int(training_set_size / batch_size)
        a,b,c,d = zip(*total_dataset)
        for i in range(num_batch):
            try:
                supp_xs = list(a[batch_size*i:batch_size*(i+1)])
                supp_ys = list(b[batch_size*i:batch_size*(i+1)])
                query_xs = list(c[batch_size*i:batch_size*(i+1)])
                query_ys = list(d[batch_size*i:batch_size*(i+1)])
            except IndexError:
                continue
            melu.global_update(supp_xs, supp_ys, query_xs, query_ys, config['inner'])

    if model_save:
        torch.save(melu.state_dict(), model_filename)

def selection(melu, master_path, topk):
    if not os.path.exists("{}/scores/".format(master_path)):
        os.mkdir("{}/scores/".format(master_path))
    if config['use_cuda']:
        melu.cuda()
    melu.eval()
    
    target_state = 'warm_state'
    dataset_size = int(len(os.listdir("{}/{}".format(master_path, target_state))) / 4)
    grad_norms = {}
    for j in list(range(dataset_size)):
        support_xs = pickle.load(open("{}/{}/supp_x_{}.pkl".format(master_path, target_state, j), "rb"))
        support_ys = pickle.load(open("{}/{}/supp_y_{}.pkl".format(master_path, target_state, j), "rb"))
        item_ids = []
        
        with open("{}/log/{}/supp_x_{}_u_m_ids.txt".format(master_path, target_state, j), "r") as f:
            for line in f.readlines():
                item_id = line.strip().split()[1]
                item_ids.append(item_id)
        
        for support_x, support_y, item_id in zip(support_xs, support_ys, item_ids):
            support_x = support_x.view(1, -1)
            support_y = support_y.view(1, -1)
            print('passed')
            norm = melu.get_weight_avg_norm(support_x, support_y, config['inner'])
            
            try:
                grad_norms[item_id]['discriminative_value'] += norm.item()
                grad_norms[item_id]['popularity_value'] += 1
            except:
                grad_norms[item_id] = {
                    'discriminative_value': norm.item(),
                    'popularity_value': 1
                }
    d_value_max = 0
    p_value_max = 0
    for item_id in grad_norms.keys():
        grad_norms[item_id]['discriminative_value'] /= grad_norms[item_id]['popularity_value']
        if grad_norms[item_id]['discriminative_value'] > d_value_max:
            d_value_max = grad_norms[item_id]['discriminative_value']
        if grad_norms[item_id]['popularity_value'] > p_value_max:
            p_value_max = grad_norms[item_id]['popularity_value']
    for item_id in grad_norms.keys():
        grad_norms[item_id]['discriminative_value'] /= float(d_value_max)
        grad_norms[item_id]['popularity_value'] /= float(p_value_max)
        grad_norms[item_id]['final_score'] = grad_norms[item_id]['discriminative_value'] * grad_norms[item_id]['popularity_value']

    movie_info = {}
    with open("./movielens/ml-1m/movies_extrainfos.dat", encoding="utf-8") as f:
        for line in f.readlines():
            tmp = line.strip().split("::")
            movie_info[tmp[0]] = "{} ({})".format(tmp[1], tmp[2])

    evidence_candidates = []
    for item_id, value in list(sorted(grad_norms.items(), key=lambda x: x[1]['final_score'], reverse=True))[:topk]:
        evidence_candidates.append((movie_info[item_id], value['final_score']))
    return evidence_candidates


if __name__ == "__main__":
    #master_path= "./ml"
    master_path = 'E:/Code/melunew/movielens/ml'
    if not os.path.exists("{}/".format(master_path)):
        os.mkdir("{}/".format(master_path))
        # ml1m preparing dataset. It needs about 22GB of your hard disk space.
        generate(master_path)

    # training model.
    melu = MeLU(config)
    model_filename = "{}/models.pkl".format(master_path)
    if not os.path.exists(model_filename):
        # Load training dataset.
        training_set_size = int(len(os.listdir("{}/warm_state".format(master_path))) / 4)
        supp_xs_s = []
        supp_ys_s = []
        query_xs_s = []
        query_ys_s = []
        for idx in range(training_set_size):
            supp_xs_s.append(pickle.load(open("{}/warm_state/supp_x_{}.pkl".format(master_path, idx), "rb")))
            supp_ys_s.append(pickle.load(open("{}/warm_state/supp_y_{}.pkl".format(master_path, idx), "rb")))
            query_xs_s.append(pickle.load(open("{}/warm_state/query_x_{}.pkl".format(master_path, idx), "rb")))
            query_ys_s.append(pickle.load(open("{}/warm_state/query_y_{}.pkl".format(master_path, idx), "rb")))
        total_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
        del(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)
        training(melu, total_dataset, batch_size=config['batch_size'], num_epoch=config['num_epoch'], model_save=True, model_filename=model_filename)
    else:
        trained_state_dict = torch.load(model_filename)
        melu.load_state_dict(trained_state_dict)

    # selecting evidence candidates.
    evidence_candidate_list = selection(melu, master_path, config['num_candidate'])
    for movie, score in evidence_candidate_list:
        print(movie, score)
