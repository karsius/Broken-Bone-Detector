clear
clc
close all

addpath(genpath('code/'));
addpath('Imgs/');
addpath('edges-master/');



img = imread(['Imgs', filesep, 'arm3.png']);
img = imrotate(img,33,'crop');

% crops the image by 30%
img = img(ceil(size(img,1)*0.3):size(img,1),ceil(size(img,2)*0.3):size(img,2));

% increases the images contrast 
img = imadjust(img); 

% this should remove pixels in a 20 radius square of the image border
%img = removeEdge(img,20);

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


% broken shit code
function imb = removeEdge(im, fsize)
    
    pad = ceil(fsize/2);
    imp = padarray(im,[pad pad],0,'both');
    
    xlen = size(im,1);
    ylen = size(im,2);
    
    for i = pad+1:(xlen-pad-1)
        for j = pad+1:(ylen-pad-1)
%             disp(i);
%             disp(j);
%             disp(i-pad);
%             disp(j-pad);
%             disp('---');
            if imp(i,j) > 20
                %disp('in if');
                chunk = imp(j-pad:j+pad,i-pad:i+pad);
                
                % if there is a value in the chunk = 0 then set rest to 0
                if find(chunk < 20)
                    %disp('here');
                    imp(j-pad:j+pad,i-pad:i+pad) = 21;
                    %disp(imp(j-pad:j+pad,i-pad:i+pad));
                end

            end
        end
    end
    
    imb = imp(pad+1:(xlen-pad-1),pad+1:(ylen-pad-1));
    %imagesc(imb);axis image;colormap gray;
end
