foregroundDetector = vision.ForegroundDetector('NumGaussians',3,'NumTrainingFrames',50);
videoReader = vision.VideoFileReader('sample.mp4');
for i = 1:150
    frame = step(videoReader); % read the next video frame
    foreground = step(foregroundDetector, frame);
end

figure; imshow(frame); title('Video Frame');
figure; imshow(foreground); title('Foreground');

se = strel('square', 3);
filteredForeground = imopen(foreground, se);
figure; imshow(filteredForeground); title('Clean Foreground');

blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', false, 'CentroidOutputPort', false, ...
    'MinimumBlobArea', 150);
bbox = step(blobAnalysis, filteredForeground);

result = insertShape(frame, 'Rectangle', bbox, 'Color', 'green');

num = size(bbox, 1);
result = insertText(result, [10 10], num, 'BoxOpacity', 1, ...
    'FontSize', 14);
figure; imshow(result); title('Detected Object');


videoPlayer = vision.VideoPlayer('Name', 'Detected Object');
videoPlayer.Position(3:4) = [650,400];  % window size: [width, height]
se = strel('square', 3); 

while ~isDone(videoReader)

    frame = step(videoReader); % read the next video frame

    % Detect the foreground in the current video frame
    foreground = step(foregroundDetector, frame);

    % Use morphological opening to remove noise in the foreground
    filteredForeground = imopen(foreground, se);

    % Detect the connected components with the specified minimum area, and
    % compute their bounding boxes
    bbox = step(blobAnalysis, filteredForeground);

    % Draw bounding boxes around the detected objects
    result = insertShape(frame, 'Rectangle', bbox, 'Color', 'green');

    % Display the number of objects found in the video frame
    num = size(bbox, 1);
    result = insertText(result, [10 10], num, 'BoxOpacity', 1, ...
        'FontSize', 14);

    step(videoPlayer, result);  % display the results
end

release(videoReader); % close the video file