
function [data, labels] = createDatastore(cell1Path, cell2Path, cell3Path, cell4Path, patchSize)
    cell1Folders = dir(fullfile(cell1Path, '*'));   
    cell1Folders = cell1Folders([cell1Folders.isdir] & ~startsWith({cell1Folders.name}, '.'));
    data = {};
    labels = [];
    for i = 1:length(cell1Folders)
        cellFolder = fullfile(cell1Path, cell1Folders(i).name);  
        pngFiles = dir(fullfile(cellFolder, '*.png')); 
        sample = [];
        for j = 1:length(pngFiles)
            img = imread(fullfile(cellFolder, pngFiles(j).name));  
            img = imresize(img, patchSize(1:2));
            sample = cat(3, sample, img);  
        end
        data = [data; {sample}];
        labels = [labels; repmat("Cell1", 1, 1)];  
    end
    cell2Folders = dir(fullfile(cell2Path, '*'));
    cell2Folders = cell2Folders([cell2Folders.isdir] & ~startsWith({cell2Folders.name}, '.'));
    for i = 1:length(cell2Folders)
        cellFolder = fullfile(cell2Path, cell2Folders(i).name);
        pngFiles = dir(fullfile(cellFolder, '*.png'));
        sample = [];
        for j = 1:length(pngFiles)
            img = imread(fullfile(cellFolder, pngFiles(j).name)); 
            img = imresize(img, patchSize(1:2)); 
            sample = cat(3, sample, img);  
        end
        data = [data; {sample}];
        labels = [labels; repmat("Cell2", 1, 1)];
    end
    cell3Folders = dir(fullfile(cell3Path, '*'));
    cell3Folders = cell3Folders([cell3Folders.isdir] & ~startsWith({cell3Folders.name}, '.'));
    for i = 1:length(cell3Folders)
        cellFolder = fullfile(cell3Path, cell3Folders(i).name);  
        pngFiles = dir(fullfile(cellFolder, '*.png')); 
        sample = [];
        for j = 1:length(pngFiles)
            img = imread(fullfile(cellFolder, pngFiles(j).name));  
            img = imresize(img, patchSize(1:2));
            sample = cat(3, sample, img); 
        end
        data = [data; {sample}];
        labels = [labels; repmat("Cell3", 1, 1)]; 
    end
    cell4Folders = dir(fullfile(cell4Path, '*'));
    cell4Folders = cell4Folders([cell4Folders.isdir] & ~startsWith({cell4Folders.name}, '.'));
    for i = 1:length(cell4Folders)
        cellFolder = fullfile(cell4Path, cell4Folders(i).name);
        pngFiles = dir(fullfile(cellFolder, '*.png')); 
        sample = [];
        for j = 1:length(pngFiles)
            img = imread(fullfile(cellFolder, pngFiles(j).name)); 
            img = imresize(img, patchSize(1:2)); 
            sample = cat(3, sample, img);  
        end
        data = [data; {sample}];
        labels = [labels; repmat("Cell4", 1, 1)]; 
    end

end
