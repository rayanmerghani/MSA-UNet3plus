import torch
import numpy as np
from monai import metrics
import matplotlib.pyplot as plt
import monai  
def compute_performance(output, label, metric, prefix=None, reduction='mean', plot_dice = True):

    binary_output = output > 0.5

    result = {}
    if 'confusion' in metric:
        confusionn = monai.metrics.get_confusion_matrix(binary_output,label, include_background=True)
        recall = monai.metrics.compute_confusion_matrix_metric("sensitivity", confusionn)
        precision = monai.metrics.compute_confusion_matrix_metric("precision", confusionn)
        f1_score = monai.metrics.compute_confusion_matrix_metric("f1_score", confusionn)
        accuracy = monai.metrics.compute_confusion_matrix_metric("accuracy", confusionn)
        specificity = monai.metrics.compute_confusion_matrix_metric("specificity", confusionn)
        

        result['recall'] = recall
        result['precision'] = precision
        result['f1'] = f1_score
        result['accuracy'] = accuracy
        result['specificity'] = specificity


    if 'dice' in metric:
        # dice
        # dice, _ = metrics.DiceMetric(reduction=reduction)(binary_output, label)
        # dice = dice.type_as(output)
        
        # dice_metric = metrics.DiceMetric(reduction="mean")
        # dice= dice_metric(binary_output, label)
        
        dice = metrics.DiceMetric(reduction=reduction)(binary_output, label)
        dice = dice.type_as(output)
        
        
        result['dice'] = dice
        
        # if plot_dice:
        #     batch_size = output.shape[0]
        #     dice_values = dice.cpu().numpy()  # Convert Dice values to numpy for plotting

        #     # Plotting Dice scores for each image in the batch
        #     plt.figure(figsize=(10, 6))
        #     plt.bar(range(batch_size), dice_values)
        #     plt.xlabel('Batch Index')
        #     plt.ylabel('Dice Score')
        #     plt.title('Dice Scores for Each Image in the Batch')
        #     plt.ylim(0, 1)  # Dice score ranges from 0 to 1
        #     plt.show()

        

    if ('asd' in metric) or ('acd' in metric):

        batch_size, n_channel = output.shape[:2]

        asd = np.empty((batch_size, n_channel))
        acd = np.empty((batch_size, n_channel))

        for b, c in np.ndindex(batch_size, n_channel):
            edges_pred, edges_gt = metrics.utils.get_mask_edges(binary_output[b,c],
                                                                label[b,c])
            asd_1 = metrics.utils.get_surface_distance(edges_pred, edges_gt)
            asd_2 = metrics.utils.get_surface_distance(edges_gt, edges_pred)

            if binary_output[b,c].sum() == 0: # failed to predict
                #asd[b,c] = np.nan
                #acd[b,c] = np.nan
                asd[b,c] = 128 * np.sqrt(2) / 2
                acd[b,c] = 128 * np.sqrt(2) / 2
            else:
                asd[b,c] = (asd_1.sum() + asd_2.sum()) / (len(asd_1) + len(asd_2))
                acd[b,c] = ((asd_1.sum()/len(asd_1)) + (asd_2.sum()/len(asd_2))) / 2

        if reduction == 'mean':
            asd = asd[~np.isnan(asd)[:,0],:].mean()
            acd = acd[~np.isnan(acd)[:,0],:].mean()

        asd = torch.from_numpy(np.array(asd)).type_as(output)
        acd = torch.from_numpy(np.array(acd)).type_as(output)

        result['asd'] = asd
        result['acd'] = acd

    if prefix:
        result = {prefix+'_'+key:value for key, value in result.items()}

    return result
