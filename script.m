load fisheriris

[~, D_pca, ~] = pca(meas);
[~,~,z] = unique(species);

plot(D_pca(z==1,1),D_pca(z==1,2),'b*')
hold on
plot(D_pca(z==2,1),D_pca(z==2,2),'g*')
plot(D_pca(z==3,1),D_pca(z==3,2),'k*')
axis equal
grid on
xlabel('Feature 1');
ylabel('Feature 2');

mdl = my_fitpca(D_pca(:,1:2),z);
plot_ellipse(mdl);

[X,Y] = meshgrid(-3:0.05:3, -2.5:0.05:2.5);
dataPoints = [X(:) Y(:)];
d = [1 1];
[class, score] = my_predictpca(mdl,dataPoints);
re = reshape(class,size(X));
contour(X,Y,re==1, [1,1],'k--', 'LineWidth', 1.1)
contour(X,Y,re==3, [1,1],'k--', 'LineWidth', 1.1)
legend('Setosa', 'Versicolor', 'Virginica','Boundary 1', 'Boundary 2');

%-----------------------------------------------------------------------%

function model = my_fitpca(Data, Classes)
    z = unique(Classes);
    numClasses = length(z);
    
    for i = 1 : numClasses
        [eigenVects,~,eigenVals,~,~,mu] = pca(Data(Classes==z(i),:));
        model.class(i).eigvects = eigenVects;
        model.class(i).eigvals = eigenVals;
        model.class(i).mu = mu;
    end
end

%-----------------------------------------------------------------------%

function plot_ellipse(model)
    numClasses = length(model.class);
    theta = 0:0.05:2*pi;
    for j = 1 : numClasses
        std_major = sqrt(model.class(j).eigvals(1));
        std_minor = sqrt(model.class(j).eigvals(2));
        x_ellipse = model.class(j).eigvects(1,1)*2*std_major*cos(theta)...
            + model.class(j).eigvects(1,2)*2*std_minor*sin(theta);
        y_ellipse = model.class(j).eigvects(2,1)*2*std_major*cos(theta)...
            + model.class(j).eigvects(2,2)*2*std_minor*sin(theta);
        plot(x_ellipse + model.class(j).mu(1), y_ellipse + ...
            model.class(j).mu(2), 'k--', 'HandleVisibility','off')
    end
end

%-----------------------------------------------------------------------%

function [dist, projection] = MahalanobisDistance(mdl,data)

    dummy = (data - mdl.mu)*mdl.eigvects;
    projection = abs(dummy)./sqrt(mdl.eigvals)';
    dist = sqrt(sum(projection.^2));
end

%-----------------------------------------------------------------------%

function [class,score] = my_predictpca(mdl,data)
    [r,~] = size(data);
    class = zeros(r,1);
    score = zeros(r,1);

    for i = 1 : r
    md = zeros(length(mdl.class),1);
        for j = 1 : length(mdl.class)
            md(j) = MahalanobisDistance(mdl.class(j),data(i,:));
        end
    [score(i), class(i)] = min(md);
    end
end
