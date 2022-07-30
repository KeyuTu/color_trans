class FeatureProcessor():
    def __init__(self, pretrain, n_splits, is_cosine_feature=False, d_feature="ed", num_classes=64,
                 preprocess_after_split="none", preprocess_before_split="none", normalize_before_center=False,
                 normalize_d=False, normalize_ed=False):
        super(FeatureProcessor, self).__init__()
        self.pretrain = pretrain
        self.feat_dim = self.pretrain.feat_dim
        self.n_splits = n_splits
        self.num_classes = 64
        self.is_cosine_feature = is_cosine_feature
        self.d_feature = d_feature
        self.preprocess_after_split = preprocess_after_split
        self.preprocess_before_split = preprocess_before_split
        self.normalize_before_center = normalize_before_center
        self.normalize_d = normalize_d
        self.normalize_ed = normalize_ed
        # print(self.preprocess_before_split, self.preprocess_after_split, self.normalize_before_center,
              # self.normalize_d, self.normalize_ed)

        pretrain_features = self.pretrain.get_pretrained_class_mean(normalize=is_cosine_feature)
        self.pretrain_features = torch.from_numpy(pretrain_features).float().cuda()[:num_classes]#64x512
        # print(pretrain_features.shape,'llllll')
        # print(normalize_d)
        
        # exit()
        #exit(0)
        if normalize_d:
            self.pretrain_features = self.normalize(self.pretrain_features)
        self.pretrain_features_mean = self.pretrain_features.mean(dim=0)#512
        # print(self.pretrain_features_mean.shape)
        # exit()
        # if self.n_splits > 1:
            # self.pretrain.load_d_specific_classifiers(n_splits)

    def get_split_features(self, x, preprocess=False, center=None, preprocess_method="l2n"):
        # Sequentially cut into n_splits parts
        #print(self.feat_dim,self.n_splits)
        split_dim = int(self.feat_dim / self.n_splits)
        split_features = torch.zeros(self.n_splits, x.shape[0], split_dim).cuda()
        #print(preprocess_method)
        #exit()
        for i in range(self.n_splits):
            start_idx = split_dim * i
            end_idx = split_dim * i + split_dim
            if preprocess_method == "l2n":
                split_features[i] = self.normalize(x[:, start_idx:end_idx])
            elif preprocess_method == "none":
                split_features[i] = x[:, start_idx:end_idx]
            elif preprocess_method == "cl2n":
                split_features[i] = self.normalize(x[:, start_idx:end_idx] - center[:, start_idx:end_idx])
            '''
            if preprocess:
                if preprocess_method != "dl2n":
                    split_features[i] = self.nn_preprocess(split_features[i], center[:, start_idx:end_idx], preprocessing=preprocess_method)
                else:
                    if self.normalize_before_center:
                        split_features[i] = self.normalize(split_features[i])
                    centered_data = split_features[i] - center[i]
                    split_features[i] = self.normalize(centered_data)
            '''
        return split_features

    def nn_preprocess(self, data, center=None, preprocessing="l2n"):
        #print(preprocessing)
        #exit()
        if preprocessing == "none":
            return data
        elif preprocessing == "l2n":
            return self.normalize(data)
        elif preprocessing == "cl2n":
            if self.normalize_before_center:
                data = self.normalize(data)
            #print(data.shape)
            #print(center.shape)
            #exit()
            centered_data = data - center
            return self.normalize(centered_data)

    def calc_pd(self, x, clf_idx):
        proba = self.pretrain.classify(x)
        return proba

    def normalize(self, x, dim=1):
        x_norm = torch.norm(x, p=2, dim=dim).unsqueeze(dim).expand_as(x)
        x_normalized = x.div(x_norm)
        return x_normalized

    def get_d_feature(self, x, x_ori):
        # print(self.feat_dim)#512
        #print(x.shape,'x.shape')
        #exit(0)
        feat_dim = int(self.feat_dim / self.n_splits)#64
        # print(feat_dim,self.d_feature) # (64, ed)
        # exit()
        if self.d_feature == "ed":
            d_feat_dim = int(self.feat_dim / self.n_splits)#64
        else:
            d_feat_dim = self.num_classes
        d_feature = torch.zeros(self.n_splits, x.shape[0], d_feat_dim).cuda()#(8,5,64)
        if x_ori is None:
            pd = self.calc_pd(x, 0)#(5,64)
        else:
            pd = self.calc_pd(x_ori, 0)
        #print(pd.shape)
        #print(pd.sum(1))
        #exit()
        for i in range(self.n_splits):
            start = i * feat_dim
            stop = start + feat_dim
            if self.d_feature == "pd":
                d_feature[i] = pd
            else:
                # print(torch.mm(pd, self.pretrain_features).shape) # torch.Size([5, 512])
                # print(pd.shape,self.pretrain_features.shape) # torch.Size([5, 64]) torch.Size([64, 512])
                # exit()
                d_feature[i] = torch.mm(pd, self.pretrain_features)[:, start:stop] # d_feature意义何在？
        return d_feature

    def get_features(self, support, query, support_ori, query_ori):
        # print(support.shape, query.shape)#(5,512)(75,512)
        # print(support_ori, query_ori)
        # exit()
        support_d = self.get_d_feature(support, support_ori)#(8,5,64)#重构特征
        query_d = self.get_d_feature(query, query_ori)#(75,5,64)
        #print(support_d.shape)
        #print(query_d.shape)
        #exit()
        if self.normalize_ed:
            support_d = self.normalize(support_d, dim=2)
            query_d = self.normalize(query_d, dim=2)
        support_size = support.shape[0]
        query_size = query.shape[0]
        #print(self.pretrain_features_mean.shape,'lll')
        #exit()
        pmean_support = self.pretrain_features_mean.expand((support_size, self.feat_dim))
        pmean_query = self.pretrain_features_mean.expand((query_size, self.feat_dim))
        support = self.nn_preprocess(support, pmean_support, preprocessing=self.preprocess_before_split)#特征归一化然后减均值
        query = self.nn_preprocess(query, pmean_query, preprocessing=self.preprocess_before_split)
        #print(self.preprocess_after_split)
        #exit()
        split_support = self.get_split_features(support, preprocess=True, center=pmean_support,
                                                preprocess_method=self.preprocess_after_split)
        split_query = self.get_split_features(query, preprocess=True, center=pmean_query,
                                              preprocess_method=self.preprocess_after_split)
        return split_support, support_d, split_query, query_d