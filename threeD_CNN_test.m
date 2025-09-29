clc
clear
close all

cell1Path = 'E:\twoD\ExpFiles\test\K562';
cell2Path = 'E:\twoD\ExpFiles\test\MEC1'; 
cell3Path = 'E:\twoD\ExpFiles\test\THP1'; 
cell4Path = 'E:\twoD\ExpFiles\test\Juarkat'; 

patchSize = [512, 512, 16]; 
[trainData_all, trainLabels_all] = createDatastore(cell1Path, cell2Path, cell3Path, cell4Path, patchSize);
trainLabels_all = categorical(trainLabels_all);
[trainData, valData, trainLabels, valLabels] = splitData(trainData_all, trainLabels_all, 0.8);
trainTable = table(trainData, trainLabels);
valTable = table(valData, valLabels);
load('D:\MyCode\matlabcode\threeDCnn\trained3DCNN.mat', 'net');  %Your trained network

trainLabels_all_Y = table(trainData_all,trainLabels_all);

YPred = classify(net, trainLabels_all_Y,'MiniBatchSize', 12);

accuracy = sum(YPred == trainLabels_all) / numel(trainLabels_all);
disp(['Validation Accuracy: ', num2str(accuracy * 100), '%']);

confusionchart(trainLabels_all, YPred,'Normalization', 'row-normalized');
