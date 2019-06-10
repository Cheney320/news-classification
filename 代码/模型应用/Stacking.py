from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
import numpy as np



class StackingModels(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # 我们将原来的模型clone出来，并且进行实现fit功能
    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # 对于每个模型，使用交叉验证的方法来训练初级学习器，并且得到次级训练集
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models) * 8))

        for i, model in enumerate(self.base_models):
            print("正在训练第{}个model".format(i + 1))
            j = 1
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                instance.fit(X[train_index], y[train_index])
                self.base_models_[i].append(instance)
                y_pred = instance.predict_proba(X[holdout_index])
                out_of_fold_predictions[holdout_index, i * 8:(i + 1) * 8] = y_pred
                print("Fold {} done".format(j))
                j += 1
        print('fit meta_model!')
        # 使用次级训练集来训练次级学习器
        self.meta_model.fit(out_of_fold_predictions, y)
        return self

    # 在上面的fit方法当中，我们已经将我们训练出来的初级学习器和次级学习器保存下来了
    # predict的时候只需要用这些学习器构造我们的次级预测数据集并且进行预测就可以了
    def predict(self, X):
        meta_features = np.zeros((X.shape[0], len(self.base_models) * 8))
        for i, models in enumerate(self.base_models_):
            fold_xtest = np.zeros((X.shape[0], 8))
            for model in models:
                fold_xtest += model.predict_proba(X)
            meta_features[:, i * 8:(i + 1) * 8] = fold_xtest / self.n_folds
        return self.meta_model.predict(meta_features), self.meta_model.predict_proba(meta_features)