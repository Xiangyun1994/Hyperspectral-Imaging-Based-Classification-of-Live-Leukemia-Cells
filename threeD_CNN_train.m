clear
clc
close all
%% Step 1: Network
layers = [
    image3dInputLayer([512 512 16 1], 'Normalization', 'none', 'Name', 'input')
   
    convolution3dLayer(3, 32, 'Padding', 'same', 'Stride', 1, 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling3dLayer([4 4 1], 'Stride', [4 4 1], 'Name', 'avgpool1')

    convolution3dLayer(3, 32, 'Padding', 'same', 'Stride', 1, 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    maxPooling3dLayer([2 2 1], 'Stride', [2 2 1], 'Name', 'maxpool1')
    dropoutLayer(0.25, 'Name', 'dropout1')

    convolution3dLayer(3, 64, 'Padding', 'same', 'Stride', 1, 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu3')
    maxPooling3dLayer([2 2 1], 'Stride', [2 2 1], 'Name', 'maxpool2')

    convolution3dLayer(3, 64, 'Padding', 'same', 'Stride', 1, 'Name', 'conv4')
    reluLayer('Name', 'relu4')
    maxPooling3dLayer([2 2 1], 'Stride', [2 2 1], 'Name', 'maxpool3')
    dropoutLayer(0.25, 'Name', 'dropout2')

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

patchSize = [512, 152, 16]; 
[trainData_all, trainLabels_all] = createDatastore(cell1Path, cell2Path, cell3Path, cell4Path, patchSize);
trainLabels_all = categorical(trainLabels_all);
[trainData, valData, trainLabels, valLabels] = splitData(trainData_all, trainLabels_all, 0.8);
trainTable = table(trainData, trainLabels);
valTable = table(valData, valLabels);

options = trainingOptions('adam', ...  
    'MaxEpochs', 10, ... 
    'InitialLearnRate', 1e-4, ... 
    'MiniBatchSize', 4, ... 
    'Shuffle', 'every-epoch', ... 
    'ValidationData', valTable, ... 
    'ValidationFrequency', 1, ... 。
    'Verbose', true, ... 
    'Plots', 'training-progress'); 
[net, trainInfo] = trainNetwork(trainTable, layers, options);
save('D:\MyCode\matlabcode\threeDCnn\trained3DCNN.mat', 'net'); %Your trained network
%% Step 3: Val 
Data_Y=valData(1:50,1);
Labels_Y=valLabels(1:320,1);

for i = 1:numel(trainData_all)
    if isnumeric(trainData_all{i})
        trainData_all_re{i} = reshape(trainData_all{i}, 512, 512, 16, 1);
    end
end

all = table(Data_Y,Labels_Y);

YPred = classify(net, all,'MiniBatchSize', 1);

accuracy = sum(YPred == trainLabels_all) / numel(trainLabels_all);
disp(['Validation Accuracy: ', num2str(accuracy * 100), '%']);

confusionchart(Labels_Y, YPred);
%% Step 4:  Loss & Accuracy Curve
trainLoss = trainInfo.TrainingLoss;
valLoss = trainInfo.ValidationLoss;

figure;
plot(trainLoss, 'b', 'LineWidth', 1.5); hold on;
plot(valLoss, 'r', 'LineWidth', 1.5);
xlabel('Epoch');
ylabel('Loss');
legend('Training Loss', 'Validation Loss');
title('Loss Curve');
grid on;

trainAccuracy = trainInfo.TrainingAccuracy;
valAccuracy = trainInfo.ValidationAccuracy;

figure;
plot(trainAccuracy, 'b', 'LineWidth', 1.5); hold on;
plot(valAccuracy, 'r', 'LineWidth', 1.5);
xlabel('Epoch');
ylabel('Accuracy (%)');
legend('Training Accuracy', 'Validation Accuracy');
title('Accuracy Curve');
grid on;

