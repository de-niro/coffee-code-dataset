import pandas as pd
import plotly.express as px
#from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn
import numpy
import copy
import os

class Evaluator():
    def __init__(self, abspath):
        self.abspath = abspath
        self.df = pd.read_csv(os.path.join(self.abspath, 'analyzer', 'coffee.csv'))
        self.var_names = ["CodingHours", "CoffeeCupsPerDay", "CupsPerHour"]
        self.preprocess_dataset()
        self.fetch_stats()
        self.configure_px()
        self.generate_graphs()
        self.run_nn_pipeline()

    def preprocess_dataset(self):
        self.df["CupsPerHour"] = self.df["CoffeeCupsPerDay"] / self.df["CodingHours"]

    def fetch_stats(self):
        self.stats = {}
        for k in self.var_names:
            val_count = self.df[k].value_counts()
            self.stats[k] = {"max": max(self.df[k]), "mean": self.df[k].mean(),\
                    "median": self.df[k].median(), "std": self.df[k].std(),\
                    "freq_i": val_count.index[0], "freq_v": val_count.iloc[0]}
            for v in self.stats[k]:
                #if type(self.stats[k][v]) == float:
                self.stats[k][v] = round(self.stats[k][v], 2)

        # Extra stats
        self.extra_stats = {}
        self.extra_stats["CoffeeTime"] = {"freq_i": self.df["CoffeeTime"].value_counts().index[0].lower().split()[0]}

    def export_graph(self, name):
        path = name + "Graph.html"
        self.graphs[name].write_html(os.path.join(self.abspath, "templates", path))
        self.graph_ext[name] = path

    def configure_px(self):
        px.defaults.color_continuous_scale = px.colors.sequential.solar
        self.color_sequence = ["#7c6f64", "#fb4934", "#b8bb26", "#fabd2f", "#83a598", "#d3869b", "#fe8019"]
        px.defaults.color_discrete_sequence = self.color_sequence

    def generate_graphs(self):
        self.graphs = {}
        self.graph_ext = {}
        # How long did coders work
        self.graphs["CodingHours"] = px.histogram(self.df["CodingHours"], title="Average coding hours",\
                labels={'value': 'Duration', 'count': 'Frequency'})
        self.graphs["CoffeeCupsPerDay"] = px.density_heatmap(self.df, x="CodingHours", y="CoffeeCupsPerDay", nbinsx=10, nbinsy=8)
        self.graphs["CoffeeCupsPerDayPie"] = px.pie(self.df, names="CoffeeCupsPerDay",\
                color_discrete_sequence=self.color_sequence)
        self.graphs["CupsPerHour"] = px.scatter(x=self.df["CodingHours"], y=self.df["CupsPerHour"],\
                color=self.df["CoffeeCupsPerDay"], labels={'x': 'CodingHours', 'y': 'CupsPerHour', 'color': 'CoffeeCupsPerDay'})
        steps = numpy.arange(0, int(max(self.df["CupsPerHour"])), 0.5)
        cups_s = self.df["CupsPerHour"].apply(lambda x: steps[numpy.abs(steps - x).argmin()])
        cups = cups_s.apply(lambda x: str(max(0, x - 0.25)) + " - " + str(x + 0.25))
        #cups = self.df[self.df.CupsPerHour.isin(self.df["CupsPerHour"].value_counts().index[:7])]["CupsPerHour"].astype(str)
        #cups = cups.append(pd.Series(["Other"]*(len(self.df["CupsPerHour"]) - len(cups))))
        self.graphs["CupsPerHourPie"] = px.pie(names=cups, color_discrete_sequence=self.color_sequence)
        self.graphs["CoffeeTime"] = px.scatter(self.df, x="CodingHours", y="CoffeeCupsPerDay", color="CoffeeTime")
        self.graphs["CoffeeTypes"] = px.scatter(self.df, x="CoffeeType", y="CoffeeCupsPerDay", title="Coffee types by cups")
        self.graphs["CoffeeTypesPie"] = px.pie(self.df, names="CoffeeType", values="CoffeeCupsPerDay", title="Coffee types by cups",\
                color_discrete_sequence=self.color_sequence)
        self.graphs["CoffeeTypesHours"] = px.pie(self.df, names="CoffeeType", values="CodingHours", title="Coffee types by coding hours",\
                color_discrete_sequence=self.color_sequence)
        self.graphs["AgeRange"] = px.density_heatmap(self.df, x="CoffeeType", y="AgeRange")

        for k in self.graphs:
            self.export_graph(k)
    
    def init_nn_dataset(self):
        df_sk = copy.deepcopy(self.df)
        #df_sk["Gender"] = df_sk["Gender"].map({'Male': 0, 'Female': 1})
        #df_sk["CoffeeTime"] = df_sk["CoffeeTime"].map({'Before coding': 0, 'While coding': 1, 'After coding': 2})
        le = LabelEncoder()
        df_sk['Gender'] = le.fit_transform(df_sk['Gender'])
        df_sk['CoffeeTime'] = le.fit_transform(df_sk['CoffeeTime'])
        df_sk['CodingWithoutCoffee'] = le.fit_transform(df_sk['CodingWithoutCoffee'])
        df_sk['CoffeeType'] = le.fit_transform(df_sk['CoffeeType'])
        df_sk['CoffeeSolveBugs'] = le.fit_transform(df_sk['CoffeeSolveBugs'])
        #df_sk['Country'] = le.fit_transform(df_sk['Country'])
        #df_sk['AgeRange'] = le.fit_transform(df_sk['AgeRange'])
        del df_sk['Country']
        del df_sk['AgeRange']

        df_sk = df_sk[df_sk.CoffeeTime.isin(df_sk["CoffeeTime"].value_counts().index[:4])]

        self.df_sk = df_sk

    def fit(self):
        self.nb = KNeighborsClassifier(n_neighbors=4)
        lenc = LabelEncoder()
        x = self.df_sk.iloc[:, :2]
        y = lenc.fit_transform(self.df_sk["CoffeeTime"])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)
        self.nb.fit(x_train, y_train)
        return x, y, (x_train, x_test, y_train, y_test)

    def accuracy(self, data):
        return roc_auc_score(data[3], self.nb.predict_proba(data[1]), multi_class='ovr')

    def parse_plot_data(self, x):
        return x.to_numpy(), pd.Index(["All the time", "Before and while coding",\
                "Before coding", "While coding"])

    def render_plot(self, x, y):
        X, target_names = self.parse_plot_data(x)
        cmap_light = ListedColormap(["cornflowerblue", "lightgreen", "orange", "cyan"])
        cmap_bold = ["darkorange", "c", "darkblue", "green"]

        h = 0.02

        for weights in ["uniform", "distance"]:
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
            Z = self.nb.predict(numpy.c_[xx.ravel(), yy.ravel()])
        
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, Z, cmap=cmap_light)
            
            # Plot also the training points
            seaborn.scatterplot(
                x=X[:, 0],
                y=X[:, 1],
                hue=target_names[y],
                palette=cmap_bold,
                alpha=1.0,
                edgecolor="black",
            )
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.title(
                "4-Class classification (k = %i, weights = '%s')" % (4, weights)
            )
            plt.xlabel("CodingHours")
            plt.ylabel("CoffeeCupsPerDay")

            plt.savefig(os.path.join(self.abspath, "static", "img", "predictor_plot.png"), optimize=True)
        return target_names

    def run_nn_pipeline(self):
        self.init_nn_dataset()
        
        x, y, train = self.fit()
        self.nn_x = x
        self.nn_y = y
        self.train = train
        
        self.acc = self.accuracy(self.train)

        self.nn_stats = {"accuracy": round(self.acc, 2)}

        self.target_names = self.render_plot(self.nn_x, self.nn_y)

    def Predict(self, coding_hours, coffee_cups):
        time = self.nb.predict([[int(coding_hours), int(coffee_cups)]])
        return self.target_names[int(time)]
