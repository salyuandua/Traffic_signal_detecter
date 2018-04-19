% 
%   [stop_HOG,stop_label]=load_data('stop',500,'matlab');
%   [light_HOG,light_label]=load_data('light',500,'matlab');
%   [yield_HOG,yield_label]=load_data('yield',500,'matlab');
%  [construction_HOG,construction_label]=load_data('construction',500,'matlab');
   [pedestrian_HOG,pedestrian_label]=load_data('pedestrian',200,'matlab');
   [speed_lim_HOG,speed_lim_label]=load_data('speed_lim',500,'matlab');
% % % %prepare training set
   X_train=[stop_HOG(1:400,:);light_HOG(1:400,:);yield_HOG(1:400,:);construction_HOG(1:400,:);...
       pedestrian_HOG(1:150,:);speed_lim_HOG(1:400,:)];
   Y_train=[stop_label(1:400,:);light_label(1:400,:);yield_label(1:400,:);...
       construction_label(1:400,:);pedestrian_label(1:150,:);speed_lim_label(1:400,:)];
% % % %prepare test set
   X_test=[stop_HOG(400:500,:);light_HOG(400:500,:);yield_HOG(400:500,:);...
       construction_HOG(400:500,:);pedestrian_HOG(150:200,:);speed_lim_HOG(400:500,:)];
   Y_test=[stop_label(400:500,:);light_label(400:500,:);yield_label(400:500,:);...
       construction_label(400:500,:);pedestrian_label(150:200,:);speed_lim_label(400:500,:)];
% % % % train model 
   model=fitcecoc(X_train,Y_train);
% %[model,X_test,Y_test]= training_model(300);
% % % %test model
   Y_predict=model.predict(X_test);
% % % %compute accuracy
   m=size(Y_predict,1);
   correct_count=0;
   for i=1:m
       if isequal(Y_predict(i),Y_test(i))
           correct_count=correct_count+1;
       end
   end
   accuracy=correct_count/m;
  fprintf('The accuracy is %d',accuracy);
 
%  %find extra pedestrain images from DataSet_HOG/extra
% [extra_HOG,extra_label]=load_data('extra',2000);
% extra_label=model.predict(extra_HOG);
% for i=1:size(extra_label,1)
%     if isequal(extra_label(i),{'pedestrian'})
%         disp(i);
%     end
%     %disp(isequal(extra_label(i),{'stop'}));
% end


%  function find_extra_ped()
%     dir_name='DataSet_HOG/extra/';
%     %read info table
%     info_table=readtable(strcat(dir_name,'info.csv'));
%     for i=1:size(info_table,1)
%         
%     end
%  end
 
 
 
 
 
 
 
% training_model2(200);
function [model]=training_model2(total_num)
 [stop_HOG,stop_label]=load_data('stop',total_num);
 [light_HOG,light_label]=load_data('light',total_num);
 [yield_HOG,yield_label]=load_data('yield',total_num);
[construction_HOG,construction_label]=load_data('construction',total_num);
  [pedestrian_HOG,pedestrian_label]=load_data('pedestrian',total_num);
  [speed_lim_HOG,speed_lim_label]=load_data('speed_lim',total_num);
% % %prepare training set
training_num=total_num*0.75;
  X_train=[stop_HOG(1:training_num,:);light_HOG(1:training_num,:);yield_HOG(1:training_num,:);...
      construction_HOG(1:training_num,:);pedestrian_HOG(1:training_num,:);speed_lim_HOG(1:training_num,:)];
  Y_train=[stop_label(1:training_num,:);light_label(1:training_num,:);yield_label(1:training_num,:);construction_label(1:training_num,:);...
      pedestrian_label(1:training_num,:);speed_lim_label(1:training_num,:)];
% % %prepare test set
  X_test=[stop_HOG(training_num:total_num,:);light_HOG(training_num:total_num,:);...
      yield_HOG(training_num:total_num,:);construction_HOG(training_num:total_num,:);...
      pedestrian_HOG(training_num:total_num,:);speed_lim_HOG(training_num:total_num,:)];
  Y_test=[stop_label(training_num:total_num,:);light_label(training_num:total_num,:);yield_label(training_num:total_num,:);...
      construction_label(training_num:total_num,:);pedestrian_label(training_num:total_num,:);speed_lim_label(training_num:total_num,:)];
% % % train model 
  model=fitcecoc(X_train,Y_train);
% % %test model
  Y_predict=model.predict(X_test);
% % %compute accuracy
  m=size(Y_predict,1);
  correct_count=0;
  for i=1:m
      if isequal(Y_predict(i),Y_test(i))
          correct_count=correct_count+1;
      end
  end
  accuracy=correct_count/m;
 fprintf('The accuracy is %d',accuracy);
end
%training model,return svm model and test set, i use 70% of data_num as 
% training data and 30% of data_num as test set
function [model,test_X,test_Y]= training_model(data_num)
    dir_names=["stop","light","yield","construction","pedestrian","speed_lim"];
    X=[];
    Y=[];
    for i=1:size(dir_names,2)
        [x,y]=load_data(dir_names(i),data_num);
        X=[X;x];
        Y=[Y;y];
    end
    %sparate X and Y
    m=size(X,1);
    thehold=m*0.7;
    %prepare training set
    training_X=X(1:int64(thehold),:);
    training_Y=cellstr(Y(1:int64(thehold),:));
    %prepare test set
    test_X=X(int64(thehold)+1:end,:);
    test_Y=cellstr(Y(int64(thehold)+1:end,:));
    %training model with svm
    model=fitcecoc(training_X,training_Y);
end

%read data method 2, 2 choices for HOG_generating_method:
%1. 'txt': read HOG from txt file directly
%2. 'matlab': extracting HOG with extractHOGFeatures function
function [HOG_vec,label_vec]=load_data(dir_name,m,HOG_generating_method)
root_dir='DataSet_images/';%use matlab function
if strcmp(HOG_generating_method,'text')%read HOG from txt files
    root_dir='DataSet_HOG/';
end

file_dir=strcat(root_dir,dir_name,'/');
 %read Y label
label_file=strcat(file_dir,'info.csv');
label_table=readtable(label_file);

[row,n]=size(label_table);
if m>row
    m=row;
end
%label vector
label_vec=cell([m,1]);
label_vec(:,:)={dir_name};
HOG_vec=zeros(m,1568);
%loop table to read HOG
for i=1:m
    %
    if strcmp(HOG_generating_method,'text')
        full_file_path=strcat(file_dir,replace(label_table.Filename(i),'ppm','txt'));
        disp(full_file_path);
        HOG=load(string(full_file_path));
        HOG_vec(i,:)=HOG';
    else
        full_file_path=strcat(file_dir,label_table.Filename(i));
        disp(full_file_path);
        img=imread(full_file_path{1});
        %convert to grey
        if size(img,3)~=1
            img=rgb2gray(img);
        end
        %resize to 40*40
        img=imresize(img,[40,40]);
        %compute HOG
        HOG=extractHOGFeatures(img,'CellSize',[5,5],'BlockSize',[2,2],'NumBins',8);
        HOG_vec(i,:)=HOG;
    end
end
end




