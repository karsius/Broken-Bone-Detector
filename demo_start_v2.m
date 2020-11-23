clear
clc
close all

addpath(genpath('code/'));
addpath('Imgs/');
addpath('edges-master/');



img = imread(['Imgs', filesep, 'arm3.png']);

img = imrotate(img,33,'crop');

% this process was under imadjust(img), and is moved up here
% this should find and blur the edge of the actual xray img and the black background
% with an average filter 20*20
img = removeEdge(img,20);
figure('Name', 'edge_removed');imagesc(img);axis image;axis off; colormap gray;

% crops the image by 30%
img = img(ceil(size(img,1)*0.3):size(img,1),ceil(size(img,2)*0.3):size(img,2));

% increases the images contrast 
img = imadjust(img); 

figure('Name', 'edge_removed');imagesc(img);axis image;axis off; colormap gray;

t = nanmedian(img,'all');
m = max(img,[],'all');
% removes 95% below the max white value
img = black2NaN(img,m*.95);
img = imresize(img, [round(size(img,1)/4), round(size(img,2)/4)]);



%compute the kernel for the image size
%you only need to compute the kernal once for one an image size
[kernels, kernels_flip, kernel_params] =kernelInitialization(img);
ticId = tic;
%the lines variable contains the detected line segmentations it arranged as
%[x1 y1 x2 y2 probability]
%The fullLines are the detected lines. It is arranged as [rho theta probability]
[lines, fullLines] =lineSegmentation_HighRes(img,kernels, kernels_flip, kernel_params);
display('Total time');
toc(ticId)
fig = figure;
imshow(img);
hold all
%Order lines by probability
lines = sortrows(lines, -5);
ttlLines = size(lines,1);
for i = 1:90
    %plot the top 90 lines
    line([lines(i,1) lines(i,3)], [lines(i,2) lines(i,4)],'Color', rand(1,3), 'LineWidth', 3);
end

%please use code in Evaluation code.zip to evaluate the performance of the line segmentation algorithm


[lines2] = mcmlsd2Algo(lines,img);
fig = figure;
imshow(img);
hold all
%Order lines by probability
lines = sortrows(lines2, -5);
ttlLines = size(lines2,1);
for i = 1:90
    %plot the top 90 lines
    line([lines2(i,1) lines2(i,3)], [lines2(i,2) lines2(i,4)],'Color', rand(1,3), 'LineWidth', 3);
end

function imout = adcontrst(im)
    im = im2double(im);
    mim = mean(reshape(im,[], 1));
    
    whitemax = max(im,[],'all','omitnan');
    
    x = size(im,1);
    y = size(im,2);
 
    msecResize = whitemax * ones(x,y);
    ratio = mean(mim)./whitemax;

    imout = (im./msecResize) .* ratio;
    imout = im2uint8(imout);
    imagesc(imout);axis image;colormap gray;
    disp('work');
end



function imb = black2NaN(im,t)
    for i = 1:size(im,1)
        for j = 1:size(im,2)
            if im(i,j) < t
                im(i,j) = 0;
            end
        end
    end
    imb = im;
end


function imb = removeEdge(im, rad)
    %works only for images from our xray dataset
    %blur the edge of anything adjacent to black pixels
    %so that the algorithm doesn't get comfused by image edges
    [row, col] = size(im);
    t = 0.01*mean(im, 'all'); %test threshold
    canvas = im;
    im = padarray(im, [rad,rad], 0);
    for r=1:row
        for c=1:col
            block = im(r:r+2*rad, c:c+2*rad);
            if min(block, [], 'all') < t
                canvas(r,c) = mean(block, 'all');
            end
        end
    end
    imb = canvas;
end
