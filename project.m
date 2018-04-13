
%[stop_HOG,stop_label]=load_data('stop',500);
%[light_HOG,light_label]=load_data('light',200);
%[yield_HOG,yield_label]=load_data('yield',200);
[construction_HOG,construction_label]=load_data('construction',200);
[pedestrian_HOG,pedestrian_label]=load_data('pedestrian',200);
[speed_lim_HOG,speed_lim_label]=load_data('speed_lim',200);
%prepare training set
X_train=[stop_HOG(1:400,:);light_HOG(1:150,:);yield_HOG(1:150,:);construction_HOG(1:150,:);pedestrian_HOG(1:150,:);speed_lim_HOG(1:150,:)];
Y_train=[stop_label(1:400,:);light_label(1:150,:);yield_label(1:150,:);construction_label(1:150,:);pedestrian_label(1:150,:);speed_lim_label(1:150,:)];
%prepare test set
X_test=[stop_HOG(400:500,:);light_HOG(150:200,:);yield_HOG(150:200,:);construction_HOG(150:200,:);pedestrian_HOG(150:200,:);speed_lim_HOG(150:200,:)];
Y_test=[stop_label(400:500,:);light_label(150:200,:);yield_label(150:200,:);construction_label(150:200,:);pedestrian_label(150:200,:);speed_lim_label(150:200,:)];
% train model 
model=fitcsvm(X_train,Y_train);
%test model
Y_predict=model.predict(X_test);
%compute accuracy
m=size(Y_predict,1);
correct_count=0;
for i=1:m
    if Y_predict(i)==Y_test(i)
        correct_count=correct_count+1;
    end
end
accuracy=correct_count/m;

%training model,return svm model and test set, i use 70% of data_num as 
% training data and 30% of data_num as test set
function [model,test_X,test_Y]= training_model(data_num)
    dir_names=["stop","light","yield","construction","pedestrian","speed_lim"];
    X,Y=[];
    for i=1:dir_names
        [x,y]=load_data(dir_names(i),data_num);
        X=[X;x];
        Y=[Y;y];
    end
end



%read data function
function [HOG_vec,label_vec]=load_data(dir_name,m)
file_dir=strcat('DataSet_HOG/',dir_name,'/');
 %read Y label
label_file=strcat(file_dir,'info.csv');
label_table=readtable(label_file);

[row,n]=size(label_table);
if m>row
    m=row;
end
label_vec=zeros(m,1);
%if they are stop signal
if label_table.ClassId(1)==14
    label_vec=ones(m,1);
end
HOG_vec=zeros(m,1568);
%loop table to read HOG
for i=1:m
    full_file_path=strcat(file_dir,replace(label_table.Filename(i),'ppm','txt'));
    disp(full_file_path);
    HOG=load(string(full_file_path));
    HOG_vec(i,:)=HOG';
end


end








%function to read all HOG in one dir
% function [HOG_vec,label_vec]= read_training_data(dir_name,label_file_name)
% %dir_name='00000/';
% %read Y label
% label_file=strcat('training_images/',dir_name,'/',label_file_name);
% label_table=readtable(label_file);
% 
% [m,n]=size(label_table);
% label_vec=zeros(m,1);
% %if they are stop signal
% if label_table.ClassId(1)==14
%     label_vec=ones(m,1);
% end
% HOG_vec=zeros(m,1568);
% %loop table to read HOG
% for i=1:m
%     full_file_path=strcat(dir_name,'/',replace(label_table.Filename(i),'ppm','txt'));
%     disp(full_file_path);
%     HOG=load(string(strcat('training_HOG/HOG_01/',full_file_path)));
%     HOG_vec(i,:)=HOG';
% end
% 
% 
% end
% %read test function
% function [HOG_vec]=read_test_data()
% %read Y label
% %label_file=strcat('training_images/',dir_name,'/',label_file_name);
% label_table=readtable('test_images/GT-final_test.test.csv');
% 
% [m,n]=size(label_table);
% %label_vec=zeros(m,1);
% %if they are stop signal
% HOG_vec=zeros(1000,1568);
% %loop table to read HOG
% for i=1:1000
%     full_file_path=replace(label_table.Filename(i),'ppm','txt');
%     %disp(string(strcat('test_HOG/HOG_01/',full_file_path)));
%      HOG=load(string(strcat('test_HOG/HOG_01/',full_file_path)));
%      HOG_vec(i,:)=HOG';
% end
% 
% end


