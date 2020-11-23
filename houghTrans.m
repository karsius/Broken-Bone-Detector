im = imread('image2.png');
im = im2gray(im);
im = imrotate(im,33,'crop');
im = im(ceil(size(im,1)*0.3):size(im,1),ceil(size(im,2)*0.3):size(im,2));
im = adcontrst(im);
[g1x,g1y,g1mag]=g1(im,2);
emap=edge(im,g1x,g1y,0.25);
out = hughdet(im, emap);


% currently I believe adcontrst is breaking things
% will need to look into it over time
% as well a lot of the functions are from class solution, these are not my code and belong to the respective lab author

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
    

function h = hughdet(im,edgem)
    

    [H, theta, rho] = hough(edgem);
    
    P = houghpeaks(H,7,'threshold',ceil(0.3*max(H(:))));
    
    lines = houghlines(im,theta,rho,P,'FillGap',2,'MinLength',60);
    figure, imshow(im), hold on
    max_len = 0;
    for k = 1:length(lines)
       xy = [lines(k).point1; lines(k).point2];
       plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
    
       % Plot beginnings and ends of lines
       plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
       plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
    
       % Determine the endpoints of the longest line segment
       len = norm(lines(k).point1 - lines(k).point2);
       if ( len > max_len)
          max_len = len;
          xy_long = xy;
       end
       
       plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','cyan');
    end
    h=1;
end

function imb = black2NaN(im,t)
    for i = 1:size(im,1)
        for j = 1:size(im,2)
            if im(i,j) < t
                im(i,j) = 120;
            end
        end
    end
    imb = im;
end

function [g1x,g1y,g1mag] = g1(im,s)
%Computes and returns the first Gaussian derivative of image at scale s 
%in x and y directions, and the gradient magnitude.  Kernels are truncated
%beyond three standard deviations of the Gaussian function.
    %im = rgb2gray(im);
    maxx = ceil(3*s);
    x = [-maxx:maxx];
    h = normpdf(x,0,s);
    im = black2NaN(im,120);
    h1 = (-x/(s^2)).*normpdf(x,0,s);
    g1x = conv2(h,h1,im,'same');
    g1y = conv2(h1,h,im,'same');
    g1mag = sqrt(g1x.^2+g1y.^2);

end

function emap = edge(im, g1x,g1y,a)
%Accepts x and y derivative maps (double format) and outputs an edge map
%(uint8 format) based upon non-maximum suppression in the gradient
%direction and thresholding at a gradient magnitude of mu + as, where mu
%and s are the mean and standard deviation of the gradient magnitude over
%the image, respectively. Edges are represented as 1, non-edges as 0.  
%
%The edge map should be the same size as the derivative maps, but no edges
%should be declared at pixels adjacent to the top, bottom, left or right of
%the image. Non-maximum suppression is based upon bilinear interpolation of
%the gradient magnitude at one-pixel displacements to either side of the
%central pixel, along the gradient direction.

    [m,n] = size(g1x);
    
    %Only compute edges for inner pixels
    g1xout = g1x(2:end-1,2:end-1);
    g1yout = g1y(2:end-1,2:end-1);
    g1mag = sqrt(g1x.^2+g1y.^2);
    g1magout = g1mag(2:end-1,2:end-1);
    [X,Y] = meshgrid([1:n],[1:m]);
    Xout = X(2:end-1,2:end-1);
    Yout = Y(2:end-1,2:end-1);
    
    %x and y components of gradient direction
    costheta = g1xout./g1magout;
    sintheta = g1yout./g1magout;
    
    %Step in positive gradient direction
    Xp = Xout + costheta;
    Yp = Yout + sintheta;
    %Step in negative gradient direction
    Xn = Xout - costheta;
    Yn = Yout - sintheta;
    
    %Estimate gradient magnitude at these locations
    g1magp = interp2(X,Y,g1mag,Xp,Yp);
    g1magn = interp2(X,Y,g1mag,Xn,Yn);
    
    mu = mean(g1mag(:));
    s = std(g1mag(:));
    
    emap = zeros(m,n);
    %Enforce threshold and non-max suppression
    emap(2:end-1,2:end-1) = g1magout>mu+a*s...
        & g1magout > g1magp & g1magout > g1magn;
    
%     figure;
%     subplot(1,2,1);
%     imagesc(im);axis image;colormap gray;
%     subplot(1,2,2);
    imagesc(emap);axis image;colormap gray;

end
