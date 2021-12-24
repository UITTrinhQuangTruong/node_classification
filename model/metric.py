import torch
import numpy as np


class Evaluator:
    def __init__(self):
        pass

    def _parse_and_check_input(self, input_dict):
        if not 'y_true' in input_dict:
            raise RuntimeError('Missing key of y_true')
        if not 'y_pred' in input_dict:
            raise RuntimeError('Missing key of y_pred')

        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

        '''
            y_true: numpy ndarray or torch tensor of shape (num_node, num_tasks)
            y_pred: numpy ndarray or torch tensor of shape (num_node, num_tasks)
        '''

        # converting to torch.Tensor to numpy on cpu
        if torch is not None and isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        if torch is not None and isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        # check type
        if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
            raise RuntimeError(
                'Arguments to Evaluator need to be either numpy ndarray or torch tensor')

        if not y_true.shape == y_pred.shape:
            raise RuntimeError('Shape of y_true and y_pred must be the same')

        if not y_true.ndim == 2:
            raise RuntimeError(
                'y_true and y_pred must to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

        # if not y_true.shape[1] == self.num_tasks:
        #     raise RuntimeError('Number of tasks for {} should be {} but {} given'.format(
        #         self.name, self.num_tasks, y_true.shape[1]))

        return y_true, y_pred

    def eval(self, input_dict):

        y_true, y_pred = self._parse_and_check_input(input_dict)
        return self._eval_acc(y_true, y_pred)

    def _eval_acc(self, y_true, y_pred):
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list.append(float(np.sum(correct))/len(correct))

        return {'acc': sum(acc_list)/len(acc_list)}


@torch.no_grad()
def accuracy_mlp(model, x, y_true, split_idx, evaluator):
    model.eval()

    out = model(x)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


@torch.no_grad()
def accuracy_sage(model, data, split_idx, evaluator):

    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc
