class StackingSGL:
    def __init__(self, groups, alpha, lbda,ind_sparse,weight = 0.5, beta=0.001, enta=0.1, max_iter_outer=100, max_iter_inner=100, rtol=1e-6):
        self.ind_sparse = numpy.array(ind_sparse) #指示哪些变量应当被推向稀疏（即变量的哪些维度应当被压缩）
        self.groups = numpy.array(groups) #变量的分组信息，用于后续的组选取和正则化
        self.alpha = alpha # 正则化项的系数，控制L1和L2正则化的强度
        self.lbda = lbda # 正则化项的系数，控制L1和L2正则化的强度
        self.beta = beta
        self.enta = enta
        self.max_iter_outer = max_iter_outer
        self.max_iter_inner = max_iter_inner
        self.rtol = rtol #收敛标准的相对容差
        self.coef_ = None   #记录第2种方法的更新权重
        self.finalcoef=None #记录最后的权重
        self.W_s = None     #记录第1种方法的更新权重
        self.weight = weight
	# weight, beta, enta: 这些参数用于调节不同部分的权重和正则化强度
	# max_iter_outer, max_iter_inner: 分别为外部和内部迭代的最大次数。

    def fit(self, X, y): #类核心
        # Assumption: group ids are between 0 and max(groups)
        # Other assumption: ind_sparse is of dimension X.shape[1] and has 0 if the dimension should not be pushed
        # towards sparsity and 1 otherwise
        n_groups = numpy.max(self.groups) + 1
        n, d = X.shape
        assert d == self.ind_sparse.shape[0]
        self.ind_sparse=numpy.ones((d,y.shape[1]))
        alpha_lambda = self.alpha * self.lbda * self.ind_sparse
        step=1
        s=0
        oldloss = 0
        df = pd.DataFrame(columns=["step", "loss"])
        XTX = numpy.dot(numpy.transpose(X), X)
        XTY = numpy.dot(numpy.transpose(X), y)
        # 1. Initialize the w 初始化权重
        Initia_w = numpy.dot(numpy.linalg.inv(XTX + self.enta * numpy.eye(d)), XTY).astype(numpy.float)
        #print("Initia_w: ",Initia_w)
        # 2. Calculate the similarity distance
        H = pairwise_distances(numpy.transpose(y), metric="cosine")
        # Calculate the step size by calculating the Lipschitz constant
        t = n /(math.sqrt(2 * math.pow((numpy.linalg.norm(XTX, ord=2)), 2) + math.pow(numpy.linalg.norm(self.beta * H, ord=2), 2)))
        #print("Lip: ",t/100000000)
        self.finalcoef = Initia_w
        self.coef_ = Initia_w
        n_samples, n_features = X.shape
        self.W_s = Initia_w
        W_s_1 = Initia_w
        Lip=math.sqrt(2*math.pow((np.linalg.norm(XTX,ord=2)),2)+math.pow(np.linalg.norm(self.beta*H ,ord=2),2))
        e=self.alpha/Lip
        #print("e: ",e)
        # Initialize b0,b1
        bk=1
        bk_1=1
        
        for iter_outer in range(self.max_iter_outer):
            # start W by the accelerate proximal gradient  使用加速近端梯度
            W_s_k = self.W_s + np.dot((bk_1 - 1) / bk ,(self.W_s - W_s_1)).astype(np.float)
            Gw_s_k = W_s_k - (1 / Lip) * ((np.dot(XTX , W_s_k) - XTY) + self.beta * np.dot(W_s_k , H))
            bk_1 = bk
            bk = (1 + math.sqrt(4 * math.pow(bk,2) + 1)) / 2
            W_s_1 = self.W_s
            # soft-thresholding operation
            #self.W_s = self.softthres(Gw_s_k,e)
            self.W_s = np.maximum(Gw_s_k-e,0)-np.maximum(-1*Gw_s_k-e,0)
            #print(self.W_s[0:5,0:5])
            # end W by the accelerate proximal gradient 
            
            beta_old = self.finalcoef.copy()
             # start W by the block coordinate descent 块坐标下降法
            for gr in range(n_groups):
                # 1- Should the group be zero-ed out?
                indices_group_k = self.groups == gr # 向量判断是否
                # Verify that the condition, w=0, is satisfied, otherwise the inner loop is entered
                if self.discard_group(X, y, indices_group_k):
                    self.coef_[indices_group_k] = 0.
                else:
                    # 2- If the group is not zero-ed out, perform GD for the group
                    beta_k = self.coef_[indices_group_k]
                    p_l = numpy.sqrt(numpy.sum(indices_group_k))
                    for iter_inner in range(self.max_iter_inner):
                        # Calculate the gradient
                        # grad_l = self._grad_l(X, y, indices_group_k)+self.beta * numpy.dot(beta_k , H)
                        grad_l = self._grad_l(X, y, indices_group_k)
                        # Update w
                        tmp = S(beta_k - t * grad_l, t * alpha_lambda[indices_group_k])
                        tmp *= numpy.maximum(1. - t * (1 - self.alpha) * self.lbda * p_l / numpy.linalg.norm(tmp), 0.)
                        if numpy.linalg.norm(tmp - beta_k) / norm_non0(tmp) < self.rtol:
                            self.coef_[indices_group_k] = tmp
                            break
                        beta_k = self.coef_[indices_group_k] = tmp
                # Calculate each group L2 loss
                s += numpy.sqrt(numpy.sum(indices_group_k)) * numpy.linalg.norm(self.coef_[indices_group_k])
            # end W by the block coordinate descent  
            self.finalcoef = self.weight * self.coef_ + (1.0-self.weight) * self.W_s
        
            # Calculate the least squares loss
            l_loss=self.unregularized_loss(X, y)
            # Calculate L2 loss
            reg_l2 = (1. - self.alpha) * self.lbda * s
            # Calculate L1 loss
            reg_l1 = numpy.linalg.norm(newdot(alpha_lambda,self.coef_), ord=1)
            # Calculated correlation
            correlation=numpy.trace(numpy.dot(H,numpy.dot(numpy.transpose(self.finalcoef),self.finalcoef)))
            # Calculate total loss
            totalloss=l_loss + reg_l2 + reg_l1+ self.beta*correlation
            df = df.append(pd.DataFrame({'step': [step], 'loss': [math.fabs(oldloss - totalloss)]}))
            step = step + 1
            if math.fabs(oldloss - totalloss) <= 0.0001:
                break
            elif totalloss <= 0:
                break
            else:
                oldloss = totalloss
            if numpy.linalg.norm(beta_old - self.finalcoef) / norm_non0(self.finalcoef) < self.rtol:
                break
        #print(df)
        #df.to_excel("stacking2_iter_loss.xls")
        return self

    def _grad_l(self, X, y, indices_group, group_zero=False):
        if group_zero:
            beta = self.coef_.copy()
            beta[indices_group.T,:] = 0.
        else:
            beta = self.coef_
        n, d = X.shape
        r = y - numpy.dot(X, beta)
        return - numpy.dot(X[:, indices_group].T, r) / n
    
    def softthres(x,e):
        a=np.maximum(x-e,0)
        b=np.maximum(-1*x-e,0)
        return a-b
    
    @staticmethod
    def _static_grad_l(X, y, indices_group, beta=None):
        n, d = X.shape
        if beta is None:
            beta = numpy.zeros((d, ))
        r = y - numpy.dot(X, beta)
        return - numpy.dot(X[:, indices_group].T, r) / n

    def unregularized_loss(self, X, y):
        n, d = X.shape
        return numpy.linalg.norm(y - numpy.dot(X, self.finalcoef)) ** 2 / (2 * n)

    def loss(self, X, y):
        alpha_lambda = self.alpha * self.lbda * self.ind_sparse
        reg_l1 = numpy.linalg.norm(alpha_lambda * self.coef_, ord=1)
        s = 0
        n_groups = numpy.max(self.groups) + 1
        for gr in range(n_groups):
            indices_group_k = self.groups == gr
            s += numpy.sqrt(numpy.sum(indices_group_k)) * numpy.linalg.norm(self.coef_[indices_group_k])
        reg_l2 = (1. - self.alpha) * self.lbda * s
        #print(reg_l1, reg_l2, self.unregularized_loss(X, y))
        return self.unregularized_loss(X, y) + reg_l2 + reg_l1

    def discard_group(self, X, y, ind):
        alpha_lambda = self.alpha * self.lbda * self.ind_sparse
        norm_2 = numpy.linalg.norm(S(self._grad_l(X, y, ind, group_zero=True), alpha_lambda[ind]))
        p_l = numpy.sqrt(numpy.sum(ind))
        return norm_2 <= (1 - self.alpha) * self.lbda * p_l

    def predict(self, X):
        return numpy.dot(X, self.coef_)

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)

    @classmethod
    def lambda_max(cls, X, y, groups, alpha, ind_sparse=None):
        n, d = X.shape
        n_groups = numpy.max(groups) + 1
        max_min_lambda = -numpy.inf
        if ind_sparse is None:
            ind_sparse = numpy.ones((d, ))
        for gr in range(n_groups):
            indices_group = groups == gr
            sqrt_p_l = numpy.sqrt(numpy.sum(indices_group))
            vec_A = numpy.abs(cls._static_grad_l(X, y, indices_group))
            if alpha > 0.:
                min_lambda = numpy.inf
                breakpoints_lambda = numpy.unique(vec_A / alpha)
                lower = 0.
                for l in breakpoints_lambda:
                    indices_nonzero = vec_A >= alpha * l
                    indices_nonzero_sparse = numpy.logical_and(indices_nonzero, ind_sparse[indices_group] > 0)
                    n_nonzero_sparse = numpy.sum(indices_nonzero_sparse)
                    a = n_nonzero_sparse * alpha ** 2 - (sqrt_p_l * (1. - alpha)) ** 2
                    b = - 2. * alpha * numpy.sum(vec_A[indices_nonzero_sparse])
                    c = numpy.sum(vec_A[indices_nonzero] ** 2)
                    delta = b ** 2 - 4 * a * c
                    if delta >= 0.:
                        candidate0 = (- b - numpy.sqrt(delta)) / (2 * a)
                        candidate1 = (- b + numpy.sqrt(delta)) / (2 * a)
                        if lower <= candidate0 <= l:
                            min_lambda = candidate0
                            break
                        elif lower <= candidate1 <= l:
                            min_lambda = candidate1
                            break
                    lower = l
            else:
                min_lambda = numpy.linalg.norm(numpy.dot(X[:, indices_group].T, y) / n) / sqrt_p_l
            if min_lambda > max_min_lambda:
                max_min_lambda = min_lambda
        return max_min_lambda

    @classmethod
    def candidate_lambdas(cls, X, y, groups, alpha, ind_sparse=None, n_lambdas=5, lambda_min_ratio=.1):
        l_max = cls.lambda_max(X, y, groups=groups, alpha=alpha, ind_sparse=ind_sparse)
        return numpy.logspace(numpy.log10(lambda_min_ratio * l_max), numpy.log10(l_max), num=n_lambdas)