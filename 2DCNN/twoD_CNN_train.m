clear
clc
close all
%% Step 1: Network
layers = [
imageInputLayer([256 256 1], 'Normalization', 'none', 'Name', 'input')
convolution2dLayer(3, 32, 'Padding', 'same', 'Stride', 1, 'Name', 'conv1')
reluLayer('Name', 'relu1')
batchNormalizationLayer('Name', 'bn1')
maxPooling2dLayer(2, 'Stride', 4, 'Name', 'avgpool1')
convolution2dLayer(3, 32, 'Padding', 'same', 'Stride', 1, 'Name', 'conv2')
reluLayer('Name', 'relu2')
maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
dropoutLayer(0.25, 'Name', 'dropout1')
convolution2dLayer(3, 64, 'Padding', 'same', 'Stride', 1, 'Name', 'conv3')
reluLayer('Name', 'relu3')
batchNormalizationLayer('Name', 'bn2')
convolution2dLayer(3, 64, 'Padding', 'same', 'Stride', 1, 'Name', 'conv4')
maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3')
fullyConnectedLayer(128, 'Name', 'fc1')
dropoutLayer(0.5, 'Name', 'dropout3')
fullyConnectedLayer(4, 'Name', 'fc2')
softmaxLayer('Name', 'softmax')
classificationLayer('Name', 'output')
];
%% Step 2: Data Prepare
cell1Path = 'E:\twoD\ExpFiles\train\K562';
cell2Path = 'E:\twoD\ExpFiles\train\MEC1';
cell3Path = 'E:\twoD\ExpFiles\train\THP1';
cell4Path = 'E:\twoD\ExpFiles\train\Juarkat';
patchSize = [256, 256, 1];
[trainData_all, trainLabels_all] = create2Datastore(cell1Path, cell2Path, cell3Path, cell4Path, patchSize);
trainLabels_all = categorical(trainLabels_all);
[trainData, valData, trainLabels, valLabels] = splitData(trainData_all, trainLabels_all, 0.8);
trainTable = table(trainData, trainLabels);valTable = table(valData, valLabels);
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 1e-4, ...
    'MiniBatchSize', 20, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', valTable, ...
    'ValidationFrequency', 1, ...
    'Verbose', true, ...
    'Plots', 'training-progress');
[net, trainInfo] = trainNetwork(trainTable, layers, options);
save('D:\MyCode\matlabcode\TwoDCnn\trained2DCNN.mat', 'net');  %Your trained network
%% Step 3: Val 
Data=valData;
Labels=valLabels;
for i = 1:numel(trainData_all)
    if isnumeric(trainData_all{i})
        trainData_all_re{i} = reshape(trainData_all{i}, 256, 256, 1, 1);
    end
end
all = table(Data,Labels);
Pred = classify(net, all);
accuracy = sum(Pred == trainLabels_all) / numel(trainLabels_all);
disp(['Validation Accuracy: ', num2str(accuracy * 100), '%']);
confusionchart(Labels, Pred,'Normalization', 'row-normalized');
confusionchart(Labels, Pred);
%% Step 4:  Loss & Accuracy Curve
trainLoss = trainInfo.TrainingLoss;
valLoss = trainInfo.ValidationLoss;
figure;
plot(trainLoss, 'b', 'LineWidth', 1.5); 
hold on;
plot(valLoss, 'r', 'LineWidth', 1.5);
xlabel('Epoch');
ylabel('Loss');
legend('Training Loss', 'Validation Loss');
title('Loss Curve');
grid on;
trainAccuracy = trainInfo.TrainingAccuracy;
valAccuracy = trainInfo.ValidationAccuracy;
figure;plot(trainAccuracy, 'b', 'LineWidth', 1.5); 
hold on;
plot(valAccuracy, 'r', 'LineWidth', 1.5);
xlabel('Epoch');
ylabel('Accuracy (%)');
legend('Training Accuracy', 'Validation Accuracy');
title('Accuracy Curve');
grid on;
