import os
import matplotlib.pyplot as plt
import torch


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.losses = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def add_loss(self, run, loss):
        assert run >= 0 and run < len(self.results)
        self.losses[run].append(loss)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

    def visualize(self, save_plot=True, output_dir='./visualize', show_plot=False):

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        plt.style.use("ggplot")
        results = torch.tensor(self.results)
        for run in range(len(results)):
            fig, axes = plt.subplots(2, 1, figsize=(8, 8))

            epochs = list(range(len(results[run])))
            train_acc, valid_acc, test_acc = results[run,
                                                     :, 0], results[run, :, 1], results[run, :, 2]

            axes[1].plot(epochs, train_acc.numpy(), label="train_acc")
            axes[1].plot(epochs, valid_acc.numpy(), label="valid_acc")
            axes[1].plot(epochs, test_acc.numpy(), label="test_acc")
            axes[0].plot(epochs, self.losses[run], label="train_loss")

            fig.suptitle("Training Loss and Accuracy")
            fig.text(0.5, 0.04, '# Epochs', ha='center', va='center')
            fig.text(0.06, 0.5, 'Accuracy / Loss', ha='center',
                     va='center', rotation='vertical')
            plt.legend(loc="lower left")

            if show_plot:
                plt.show()

            if save_plot:
                output_path = os.path.join(
                    output_dir, f"run_{run}.png")
                plt.savefig(output_path, bbox_inches='tight', dpi=150)

                print(f'Save plot in {output_path}')
