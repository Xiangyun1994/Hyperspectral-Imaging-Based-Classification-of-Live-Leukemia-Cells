function [data, labels] = create2Datastore(cell1Path, cell2Path, cell3Path, cell4Path, patchSize)
    data = {};
    labels = [];
    pngFiles = dir(fullfile(cell1Path, '*.png')); 
    for i = 1:length(pngFiles)
        img = imread(fullfile(cell1Path, pngFiles(i).name)); 
        img = imresize(img, patchSize(1:2));
        data = [data; {img}]; 
        labels = [labels; "Cell1"]; 
    end
    pngFiles = dir(fullfile(cell2Path, '*.png'));
    for i = 1:length(pngFiles)
        img = imread(fullfile(cell2Path, pngFiles(i).name));
        img = imresize(img, patchSize(1:2));
        data = [data; {img}];
        labels = [labels; "Cell2"];
    end
    pngFiles = dir(fullfile(cell3Path, '*.png'));
    for i = 1:length(pngFiles)
        img = imread(fullfile(cell3Path, pngFiles(i).name));
        img = imresize(img, patchSize(1:2));
        data = [data; {img}];
        labels = [labels; "Cell3"];
    end
    pngFiles = dir(fullfile(cell4Path, '*.png'));
    for i = 1:length(pngFiles)
        img = imread(fullfile(cell4Path, pngFiles(i).name));
        img = imresize(img, patchSize(1:2));
        data = [data; {img}];
        labels = [labels; "Cell4"];
    end
