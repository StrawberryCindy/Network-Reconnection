function method2016()
    clear;
    clc;%%Ѱ�ҽ�����
    
    %% ��������ͼ
    xm = 100; %�����곤��
    ym = 100;  %�����곤��
    sink.x = xm/2; %��վ������
    sink.y = ym-10; %��վ������
    density = 1/50;  %�ڵ��ܶ�
    n = xm*ym*density;  %�ڵ����
    Eo = 0.5;   %��ʼ����
    Efs = 10*0.000000000001; %���ɿռ�ģ�ͷ���һ�������ݹ�����������
    Emp = 0.0013*0.000000000001; %�ྶ˥��ģ�͹���
    d_to_BS = (abs(sink.y-ym)+sqrt(sink.x^2+sink.y^2))/2;  %�㵽��վ�ľ���
    n_CH_opt = sqrt(n/(2*pi))*sqrt(Efs/Emp)*sqrt(xm*ym)/(d_to_BS^2);
    R = sqrt(xm*ym/(n_CH_opt*pi));   %�ڵ�ͨ�Ű뾶
    %%%%%%%%%%%%%%%%%%%%%%%%% END OF PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:n
        S(i).xd = rand(1,1)*xm;%����
        S(i).yd = rand(1,1)*ym;
        S(i).G = 0; %=0��ʾ���ʸ��Ϊ��ͷ
        S(i).cl = 0;%��Ϊ��ͷ�Ĵ���
        S(i).type = 'N';%��ͨ�ڵ�
        S(i).E = Eo;
        S(i).Dis = 'n';
    end
    S(n+1).xd = sink.x;
    S(n+1).yd = sink.y;
    figure(1);
    subplot(1,2,1)
    for i = 1:n
        plot(S(i).xd,S(i).yd,'k.');
        hold on;
    end
    plot(S(n+1).xd,S(n+1).yd,'rp');  %% ����sink��
    text(S(n+1).xd,S(n+1).yd,' sink','Color','red');
    ylabel('Y-axis (m)','fontsize',10);
    xlabel('X-axis (m)','fontsize',10);
    axis equal
    axis([0 100 0 100]);%%����ͼ�Ĵ�С
    title('simulation&demo');
    
    %% ����ƻ�һЩ��
    Unique_num = 10 ;  %%�ƻ���ĸ���
    Unique_node = 100*rand(Unique_num,2);
    CH=[10 20 30 40 50 60 70 80 90 100];
    %Unique_num = length(CH);
    temp = [];
    temp1 = [];
    for i=1:length(CH)
        temp = [temp S(CH(i)).xd];
        temp1 = [temp1 S(CH(i)).yd];
    end
    Unique_node = [temp' temp1'];
    Unique_node_x = Unique_node(:,1);
    Unique_node_y = Unique_node(:,2);
    hold on;
    for i = 1:Unique_num
        theta = 0:pi/20:2*pi;
        Circle1 = Unique_node(i,1)+R*cos(theta);
        Circle2 = Unique_node(i,2)+R*sin(theta);
        % plot(Circle1,Circle2,'g-');   % ����ͨ�Ű뾶
        hold on
    end
    subplot(1,2,2)
    for i = 1:n
        plot(S(i).xd,S(i).yd,'k.');
        hold on;
    end
    plot(S(n+1).xd,S(n+1).yd,'rp');  %% ����sink��
    text(S(n+1).xd,S(n+1).yd,' sink','Color','red');
    ylabel('Y-axis (m)','fontsize',10);
    xlabel('X-axis (m)','fontsize',10);
    axis equal
    hold on
    plot(Unique_node_x,Unique_node_y,'bx');
    axis([0 100 0 100]);%%����ͼ�Ĵ�С
    hold on
    %% �������֮��ľ���(���������С��2*R�����߱��ڹ۲�)
    Unique_dis = zeros(Unique_num-1,Unique_num);
    for ii = 1:Unique_num
        for jj = ii+1:Unique_num
            Unique_dis(ii,jj) = sqrt((Unique_node_x(ii)-Unique_node_x(jj)).^2+(Unique_node_y(ii)-Unique_node_y(jj)).^2);
            if(Unique_dis(ii,jj)<=2*R)
                % plot([Unique_node_x(ii) Unique_node_x(jj)],[Unique_node_y(ii) Unique_node_y(jj)],'b-');
                hold on
            end
        end
    end
    %%%��ȡ��ͨ���x��y����
    Ordinary_node_x = cat(1,S.xd);
    Ordinary_node_y = cat(1,S.yd);
    Ordinary_node = [Ordinary_node_x Ordinary_node_y]';
    for i=1:length(CH)
        temp = ismember(Ordinary_node, [S(CH(i)).xd;S(CH(i)).yd])*(-1)+1;
        Ordinary_node = Ordinary_node.*temp;
    end
    Ordinary_num = 200;
    %% �����ж�
    if(Unique_num == 0)
        disp('      û�������! ���޸Ĳ�����');
    end
    % %%��������Ϊ1�����
    % UandO_dis = zeros(1,Ordinary_num);
    % U1_x = [];U1_y = []; U1_UO_dis = [];
    % if(Unique_num == 1)
    %     for jj = 1:Ordinary_num
    %         UandO_dis(jj) = sqrt((Ordinary_node_x(jj)-Unique_node(1,1)).^2+(Ordinary_node_y(jj)-Unique_node(1,2)).^2);
    %         if( UandO_dis(jj) <= R)
    %             U1_x = [U1_x Ordinary_node_x(jj)];
    %             U1_y = [U1_y Ordinary_node_y(jj)];
    %             U1_UO_dis = [U1_UO_dis UandO_dis(jj)];
    %         end
    %     end
    %     %%U1�еĵ�ļ���
    %     U1 = [U1_x;U1_y;U1_UO_dis];%%��һ��Ϊ�����꣬�ڶ���Ϊ������,������Ϊ��ͨ���������֮��ľ���
    % %    fprintf('��ֻ��һ�������U1ʱ����%d������Բ�ڻ�Բ�ϣ�\n',length(U1));
    % %     return;
    %     U1=  U1'
    % end
    %% �����ĸ������ڵ���1���������ʵ��������Ϊ1�������
    %if(Unique_num>=1)
    if(Unique_num>=1)
        for i = 1:Unique_num
            for k = 1:Ordinary_num
                UandO_dis(i,k) = sqrt((Ordinary_node_x(k)-Unique_node(i,1)).^2+(Ordinary_node_y(k)-Unique_node(i,2)).^2);
                if( UandO_dis(i,k)<=R)
                    U(i,k) = k;
                else
                    U(i,k) = 0;
                end
            end
        end
    end

    %% ����ÿ����ͨ����������ĸ���num������Intersection �����������ڵĵ�IntersectionPoints
    %%%   Intersection ָ����Unique_num����������ͨ��ľ���ֲ��������磺��һ�еĵڶ��е�ֵΪ0�������2����ͨ�㲻��U1�ڡ�
    Intersection = [];
    IntersectionPoints = [];
    Intersection_1 = [];
    IntersectionPoints_1 = [];
    Intersection_0 = [];
    IntersectionPoints_0 = [];
    for k = 1:Ordinary_num
        AllNodes_num(k) = length(find(U(1:Unique_num,k) ~= 0));
        if(length(find(U(1:Unique_num,k))) == 0)
            Intersection_0 = [Intersection_0  k];
            IntersectionPoints_0 = [IntersectionPoints_0 Ordinary_node(:,k)];
        end
        if(length(find(U(1:Unique_num,k))) == 1)
            Intersection_1 = [ Intersection_1 U(1:Unique_num,k)];
            IntersectionPoints_1 = [IntersectionPoints_1 Ordinary_node(:,k)];
        end
        if(length(find(U(1:Unique_num,k))) >= 2)
            Intersection = [ Intersection U(1:Unique_num,k)];
            IntersectionPoints = [IntersectionPoints Ordinary_node(:,k)];
        end
    end
    %ɾ���ظ������
    for i=1:length(CH)
        [temp1 temp2]=find(Intersection==CH(i));
        Intersection(:,temp2)=[];
    end

    for i=1:length(CH)
        [temp1 temp2]=find(Intersection_0==CH(i));
        Intersection_0(:,temp2)=[];
    end

    for i=1:length(CH)
        [temp1 temp2]=find(Intersection_1==CH(i));
        Intersection_1(:,temp2)=[];
    end


    %% ����Intersection������Ϊһ�������ͬʱ�ڶ������
    %                 for ii = 1:length(Intersection)
    %                     if(length(find(Intersection(:,ii)~=0)) >= 3)
    %                         [Final_Intersection{ii}] = GetSubsetAndMerge(Intersection(:,ii));
    %                     end
    %                 end
    %                 UU=[];
    %                 for i = 1:length(Final_Intersection)
    %                     idx = cellfun(@(x)~isempty(x),Final_Intersection,'UniformOutput',true);
    %                     if(idx(i) == 1)
    %                         UU = [UU Final_Intersection{i}{1}];
    %                     end
    %                 end
    %                 Intersection = [unique([Intersection UU]','rows')]';%%ɾ��Intersection�е��ظ���
    %% �����������

    if(Unique_num == 1)
        FinalNullAreaNodes_cell = {};
        FinalOneAreaNodes_cell = cell(2,Unique_num);
        FinalNullAreaNodes_cell{1,1} = 0;
        FinalNullAreaNodes_cell{2,1} = Intersection_0';
        for i = 1:Unique_num
            FinalOneAreaNodes_cell{1,i} = i ;%����������ڵ�λ��
            for j =1:length(Intersection_1(1,:))
                if(Intersection_1(i,j) ~= 0)
                    FinalOneAreaNodes_cell{2,i} = [FinalOneAreaNodes_cell{2,i} Intersection_1(i,j)];%��Ӧ������ڵ���ͨ��
                end
            end
        end
        Final_Cell = cat(2,cat(2, FinalNullAreaNodes_cell, FinalOneAreaNodes_cell));

    elseif(Unique_num >=2 && length(Intersection)==0)
        FinalNullAreaNodes_cell = {};
        FinalOneAreaNodes_cell = cell(2,Unique_num);
        FinalNullAreaNodes_cell{1,1} = 0;
        FinalNullAreaNodes_cell{2,1} = Intersection_0';
        FinalOneAreaNodes_cell = cell(2,Unique_num);
        for i = 1:Unique_num
            FinalOneAreaNodes_cell{1,i} = i ;%����������ڵ�λ��
            for j =1:length(Intersection_1(1,:))
                if(Intersection_1(i,j) ~= 0)
                    FinalOneAreaNodes_cell{2,i} = [FinalOneAreaNodes_cell{2,i} Intersection_1(i,j)];%��Ӧ������ڵ���ͨ��
                end
            end
        end
        Final_Cell = cat(2,cat(2, FinalNullAreaNodes_cell, FinalOneAreaNodes_cell));

    elseif(length(Intersection)~=0)
        Points = [];
        if (length(Intersection)>0)
            Point_cell = cell(1,length(Intersection(1,:)));
            for j = 1:length(Intersection(1,:))  %�޸�Intersection_1
                Points(j) = max(sort(Intersection(1:Unique_num,j)));
                Point_cell{1,j} = find(Intersection(1:Unique_num,j) ~= 0);
            end
        end


        %%Ԫ������ɾ���ظ���
        [~,k] = unique(cellfun(@char,cellfun(@getByteStreamFromArray,Point_cell,'un',0),'un',0));
        IntersectionArea = Point_cell(k);

        %%ȷ���ཻ����ÿ��Ԫ���е���ͨ����
        FinalMultiAreaNodes_cell = cell(2,length(IntersectionArea));
        for i=1:Unique_num
            for j =1:length(Intersection(1,:))
                for k=1:length(IntersectionArea)
                    FinalMultiAreaNodes_cell{1,k} = IntersectionArea{1,k};
                    if(length(find(Intersection(:,j)))==length(IntersectionArea{k}))
                        if(find(Intersection(:,j))==IntersectionArea{k})
                            if(Intersection(i,j) ~= 0)
                                FinalMultiAreaNodes_cell{2,k} = unique([FinalMultiAreaNodes_cell{2,k} Intersection(i,j)]);
                            end
                        end
                    end
                end
            end
        end


        %% ���طǽ������������Ľڵ���
        %%%�����������������ͨ�㣬����������������������������ͨ�㣻�㶼��������������ཻ��Χ�ڡ�
        FinalOneAreaNodes_cell = cell(2,Unique_num);
        for i = 1:Unique_num
            FinalOneAreaNodes_cell{1,i} = i ;%����������ڵ�λ��
            for j =1:length(Intersection_1(1,:))
                if(Intersection_1(i,j) ~= 0)
                    FinalOneAreaNodes_cell{2,i} = [FinalOneAreaNodes_cell{2,i} Intersection_1(i,j)];%��Ӧ������ڵ���ͨ��
                end
            end
        end
        %% ���ط������ڵĵ���Ϣ�Լ����ֽڵ�ĸ������
        FinalNullAreaNodes_cell = {};
        FinalNullAreaNodes_cell{1,1} = 0;
        FinalNullAreaNodes_cell{2,1} = Intersection_0';
        %     fprintf('�ֲ����ظ�������������ͨ�������� %d ����\n',n-length(Intersection_0)-length(Intersection_1));
        %     fprintf('�ֲ��ڷ��ظ�������������ͨ�������� %d ����\n',length(Intersection_1));
        %     fprintf('�ֲ��ڷ�������������ͨ�������� %d ����\n',length(Intersection_0));
        %% ����ÿ��������������ͨ�����
        Final_Intersection = [Intersection Intersection_1];
        Final_IntersectionCell = cell(2,Unique_num);
        for i = 1:Unique_num
            Final_IntersectionCell{1,i} = i;
            temp = Final_Intersection(i,:);
            temp(temp==0)=[];
            temp = unique(temp);
            Final_IntersectionCell{2,i} =  temp;
        end
        Final_Cell = cat(2,cat(2, FinalNullAreaNodes_cell, FinalOneAreaNodes_cell),FinalMultiAreaNodes_cell);
    end
    Final_Cell{2,1} = Final_Cell{2,1}';
    AA=[];
    for i=1:length(Final_Cell(1,:))
        AA=cat(2,AA,Final_Cell{2,i});
    end
    BB=unique(AA);
    %% ���۲����������������
%     figure(2);
%     for i = 1:length(Final_Cell{2,1})
%         temp = Intersection_0(i);
%         plot(S(temp).xd,S(temp).yd,'k.');
%         hold on;
%     end
%     %plot(S(n+1).xd,S(n+1).yd,'bp');
%     ylabel('Y-axis (m)','fontsize',10);
%     xlabel('X-axis (m)','fontsize',10);
%     title('simulation&demo');
%     hold on
%     axis([0 100 0 100]);%%����ͼ�Ĵ�С
%     hold on
%     for i = 1:Unique_num
%         theta = 0:pi/20:2*pi;
%         Circle1 = Unique_node(i,1)+R*cos(theta);
%         Circle2 = Unique_node(i,2)+R*sin(theta);
%         plot(Circle1,Circle2,'g-');
%         hold on
%     end
%     axis equal
%     for ii = 1:Unique_num
%         text(Unique_node_x(ii)+1,Unique_node_y(ii),strcat('U',num2str(ii)),'fontsize',10);
%     end
%     hold on
%     plot(Unique_node_x,Unique_node_y,'rp');
%     axis([0 100 0 100]);%%����ͼ�Ĵ�С
%     hold on
    
    %% �������Ҽ�������
%     %%��һ�����ҳ��߽��
%     redNodes = [];
%     for i = 1:length(CH)
%         for j = 1:length(Final_Cell{2,1})
%             temp = Intersection_0(j);
%             O_U_Dis(i,j) = sqrt((Unique_node_x(i)-S(Intersection_0(j)).xd)^2+(S(Intersection_0(j)).yd-Unique_node_y(i))^2);
%             if( O_U_Dis(i,j)<R+5)
%                 plot(S(temp).xd,S(temp).yd,'r.');
%                 hold on;
%                 redNodes = [redNodes Intersection_0(j)];%%��ɫ�߽�����
%             end
%         end
%     end
%     for i = 1:length(Final_Cell{2,1})
%         Target_A01_x0(i) = Ordinary_node(1,Intersection_0(i));
%         Target_A01_y0(i) = Ordinary_node(2,Intersection_0(i));
%     end
%     Target_A01_x0 = Target_A01_x0';
%     Target_A01_y0 = Target_A01_y0';
    %% ͹���㷨
%     DT_A01 = delaunayTriangulation(Target_A01_x0,Target_A01_y0);
%     k_A01 = convexHull(DT_A01);
%     figure(3);
%     plot(DT_A01.Points(:,1),DT_A01.Points(:,2),'g.','markersize',10);
%     hold on
%     plot(DT_A01.Points(k_A01,1),DT_A01.Points(k_A01,2),'r');
%     axis([0 100 0 100]);%%����ͼ�Ĵ�С
%     hold on

end