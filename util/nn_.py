import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_products, num_categories, num_urls, embedding_dim, hidden_dim, dropout_rate, loss=True,
                 lag_targets=None, lag_categories=None, lag_urls=None, low_targets=None):
        super().__init__()
        self.loss = loss
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=low_targets)

        def add_embedding(key, dim):
            setattr(self, 'embedding_' + key, nn.Embedding(dim, self.embedding_dim))
            getattr(self, 'embedding_' + key).weight.data.normal_(0., 0.01)

        def add_linear(key, in_feature=1):
            setattr(self, 'linear_' + key, nn.Linear(in_feature, self.embedding_dim, bias=False))
            setattr(self, 'norm_' + key, nn.BatchNorm1d(self.embedding_dim))
            setattr(self, 'activation_' + key, nn.ReLU())

        self.dict_embed = {'dayofweek': 7,
                           'hour': 24,
                           'weekend': 2,
                           'product_action_id': 4,
                           'price_null': 2,
                           'description_null': 2,
                           'img_null': 2,
                           }
        self.list_linear = ['last_event_length',
                            'cum_product',
                            'cum_search',
                            'cum_pageview',
                            'cum_event',
                            'num_search',
                            'num_pageview',
                            'num_following_search',
                            'num_following_pageview',
                            'cum_search_last',
                            'cum_pageview_last',
                            'lapse',
                            'product_count',
                            'category_hash_count',
                            'hashed_url_count',
                            'price_bucket']
        add_embedding('products', num_products)
        add_embedding('categories', num_categories)
        add_embedding('urls', num_urls)
        add_linear('lag_descriptions', 8)
        add_linear('lag_imgs', 8)
        for k, v in self.dict_embed.items():
            add_embedding(k, v)
        for k in self.list_linear:
            add_linear(k)
        print('product embedding data shape', self.embedding_products.weight.shape)
        print('category embedding data shape', self.embedding_categories.weight.shape)

        self.linear1 = nn.Linear((len(lag_targets)+len(lag_categories)+len(lag_urls)+1)*embedding_dim, hidden_dim)
        self.norm_1 = nn.BatchNorm1d(hidden_dim)
        self.activation_1 = nn.PReLU()
        self.dropout_1 = nn.Dropout(self.dropout_rate)
        self.linear_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm_2 = nn.BatchNorm1d(self.hidden_dim)
        self.activation_2 = nn.PReLU()
        self.dropout_2 = nn.Dropout(self.dropout_rate)
        self.linear_3 = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.norm_3 = nn.BatchNorm1d(self.embedding_dim)
        self.activation_3 = nn.PReLU()
        self.dropout_3 = nn.Dropout(self.dropout_rate)
        self.output_layer_bias = nn.Parameter(torch.Tensor(num_products)).data.normal_(0., 0.01)

    def get_embed(self, x, embed):
        batch_size = x.shape[0]
        x = embed(x)
        return x.view(batch_size, -1)

    def get_linear_embedding(self, key, input_dict):
        activation = getattr(self, 'activation_' + key)
        norm = getattr(self, 'norm_' + key)
        linear = getattr(self, 'linear_' + key)
        return activation(norm(linear(input_dict[key])))

    def forward(self, input_dict):
        lag_embed = self.get_embed(input_dict['lag_products'], self.embedding_products)
        first_product_embed = self.get_embed(input_dict['first_product'], self.embedding_products)
        category_embed = self.get_embed(input_dict['lag_categories'], self.embedding_categories)
        first_category_embed = self.get_embed(input_dict['first_category'], self.embedding_categories)
        url_embed = self.get_embed(input_dict['lag_urls'], self.embedding_urls)
        first_url_embed = self.get_embed(input_dict['first_url'], self.embedding_urls)
        x = (first_product_embed + first_category_embed + first_url_embed)

        for k in self.dict_embed.keys():
            x += self.get_embed(input_dict[k], getattr(self, 'embedding_' + k))

        for k in self.list_linear:
            x += self.get_linear_embedding(k, input_dict)
        x += self.get_linear_embedding('lag_descriptions', input_dict)
        x += self.get_linear_embedding('lag_imgs', input_dict)

        x = torch.cat([lag_embed, category_embed, url_embed, x], -1)
        x = self.activation_1(self.norm_1(self.linear_1(x)))
        x = self.dropout_1(x)
        x = self.activation_2(self.norm_2(self.linear_2(x)))
        x = self.dropout_2(x)
        x = self.activation_3(self.norm_3(self.linear_3(x)))
        x = self.dropout_3(x)
        logits = F.linear(x, self.embedding_products.weight, bias=self.output_layer_bias)
        output_dict = {'logits': logits}
        if self.loss:
            target = input_dict['target'].squeeze(1)
            output_dict['loss'] = self.loss_func(logits, target)
        return output_dict


class PageviewMLP(nn.Module):
    def __init__(self, num_products, num_urls, embedding_dim, hidden_dim, dropout_rate,
                 loss=True, lag_urls=None, low_targets=None):
        super().__init__()
        self.loss = loss
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=low_targets)

        def add_embedding(key, dim):
            setattr(self, 'embedding_' + key, nn.Embedding(dim, self.embedding_dim))
            getattr(self, 'embedding_' + key).weight.data.normal_(0., 0.01)

        def add_linear(key):
            setattr(self, 'linear_' + key, nn.Linear(1, self.embedding_dim, bias=False))
            setattr(self, 'norm_' + key, nn.BatchNorm1d(self.embedding_dim))
            setattr(self, 'activation_' + key, nn.ReLU())

        self.dict_embed = {'dayofweek': 7,
                           'hour': 24,
                           'weekend': 2,
                           }
        self.list_linear = ['last_event_length',
                            'hashed_url_count',
                            'count_search',
                            'count_pageview',
                            'num_search',
                            'lapse'
                            ]

        add_embedding('urls', num_urls)
        for k, v in self.dict_embed.items():
            add_embedding(k, v)
        for k in self.list_linear:
            add_linear(k)
        print('product embedding url shape: ', self.embedding_urls.weight.shape)

        self.linear_1 = nn.Linear((len(lag_urls)+1)*embedding_dim, hidden_dim)
        self.norm_1 = nn.BatchNorm1d(hidden_dim)
        self.activation_1 = nn.PReLU()
        self.dropout_1 = nn.Dropout(self.dropout_rate)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm_2 = nn.BatchNorm1d(hidden_dim)
        self.activation_2 = nn.PReLU()
        self.dropout_2 = nn.Dropout(self.dropout_rate)
        self.linear_3 = nn.Linear(hidden_dim, num_products)
        self.norm_3 = nn.BatchNorm1d(num_products)
        self.activation_3 = nn.PReLU()

    def get_embed(self, x, embed):
        batch_size = x.shape[0]
        x = embed(x)
        return x.view(batch_size, -1)

    def get_linear_embedding(self, key, input_dict):
        activation = getattr(self, 'activation_' + key)
        norm = getattr(self, 'norm_' + key)
        linear = getattr(self, 'linear_' + key)
        return activation(norm(linear(input_dict[key])))

    def forward(self, input_dict):
        lag_embed = self.get_embed(input_dict['lag_urls'], self.embedding_urls)
        first_url_embed = self.get_embed(input_dict['first_url'], self.embedding_urls)
        x = (first_url_embed)

        for k in self.dict_embed.keys():
            x += self.get_embed(input_dict[k], getattr(self, 'embedding_' + k))

        for k in self.list_linear:
            x += self.get_linear_embedding(k, input_dict)

        x = torch.cat([lag_embed, x], -1)
        x = self.activation_1(self.norm_1(self.linear_1(x)))
        x = self.dropout_1(x)
        x = self.activation_2(self.norm_2(self.linear_2(x)))
        x = self.dropout_2(x)
        logits = self.activation_3(self.norm_3(self.linear_3(x)))
        output_dict = {'logits': logits}
        if self.loss:
            target = input_dict['target'].squeeze(1)
            output_dict['loss'] = self.loss_func(logits, target)
        return output_dict
