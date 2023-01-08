import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import pycountry
import dash_bootstrap_components as dbc

## Get each country 3 letter code
def get_alpha_3(location):
    try:
        return pycountry.countries.get(name=location).alpha_3
    except:
        return 'None'

app = Dash(__name__,external_stylesheets=[dbc.themes.COSMO])
df = pd.read_csv('./netflix-rotten-tomatoes-metacritic-imdb.csv')

## Death Note has two movies with the same title, and messed up charts
df.loc[6834,'Title'] = 'Death Note (1)'
df.loc[9589,'Title'] = 'Death Note (2)'

## Get count of movies and series and store it on dataframe
p = df['Series or Movie'].value_counts()
pie_i = p.to_frame().reset_index()
pie_i.rename(columns={'index':'type','Series or Movie':'count'},inplace=True)

## Split language column and get the count of each language to store it on dataframe
df['Languages'] = df['Languages'].str.replace(', ',',')
l = df['Languages'].str.split(',',expand = True).stack().value_counts()
languages = l.to_frame().reset_index()
languages.rename(columns={'index':'language',0:'count'},inplace=True)

## Split countries column and apply the function declared earlier, to get 3 letter code to find on choropleth graph
c = df['Country Availability'].str.split(',',expand = True).stack().value_counts()
countries = c.to_frame().reset_index()
countries.rename(columns = {'index':'name',0:'count'},inplace = True)
countries['name'] = countries['name'].replace({'Russia':'Russian Federation','Czech Republic':'Czechia','South Korea':'Korea, Republic of'})
countries['code'] = countries['name'].apply(lambda x: get_alpha_3(x))

## Group content by release date and reconvert to dataframe
df['Netflix Release Date'] = pd.to_datetime(df['Netflix Release Date'])
df['year'] = df['Netflix Release Date'].dt.year
years_groups = df.groupby('year')['Series or Movie'].value_counts()
y = years_groups.unstack(level=1).reset_index()
years = list(y['year'])
years_m = list(y['Movie'])
years_s = list(y['Series'])

## Split genres column and get the number of occurences of each genres, to store it on dataframe
df['Genre'] = df['Genre'].str.replace(', ',',')
g = df['Genre'].str.split(',',expand = True).stack().value_counts()
genres = g.to_frame().reset_index()

## Get top 15 directors with most occurences, and then their average movie or series IMDb score
d = df['Director'].value_counts().head(15)
directors = d.to_frame().reset_index()
directors.rename(columns={'index':'director','Director':'count'},inplace=True)
d_avgs = []
for d in directors['director']:
    avg = df.loc[df['Director'] == d]['IMDb Score'].mean()
    d_avgs.append(avg)
directors['avg'] = d_avgs

## Get top 15 writers with most occurences, and then their average movie or series IMDb score
w = df['Writer'].value_counts().head(15)
writers = w.to_frame().reset_index()
writers.rename(columns={'index':'writer','Writer':'count'},inplace=True)
w_avgs = []
for w in writers['writer']:
    avg = df.loc[df['Writer'] == w]['IMDb Score'].mean()
    w_avgs.append(avg)
writers['avg'] = w_avgs

## Get top 15 actors/actresses with most occurences, and then their average movie or series IMDb score
df['Actors'] = df['Actors'].str.replace(', ',',')
a = df['Actors'].str.split(',',expand = True).stack().value_counts().head(15)
actors = a.to_frame().reset_index()
actors.rename(columns={'index':'actor',0:'count'},inplace=True)
a_avgs = []
n = df[df['Actors'].notna()]
for a in actors['actor']:
    avg = n.loc[n['Actors'].str.contains(a)]['IMDb Score'].mean()
    a_avgs.append(avg)
actors['avg'] = a_avgs

## Movies and series pie chart
pie_1 = px.pie(pie_i,values='count',names='type')
pie_1.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

## Languages pie chart
pie_2 = px.pie(languages,values='count',names='language')
pie_2.update_traces(textposition='inside')
pie_2.update_layout(uniformtext_minsize=12, uniformtext_mode='hide',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')

## Release dates chart
fig3 = go.Figure(
    data = [
        go.Bar(name='Movies', x=years, y=years_m, yaxis='y', offsetgroup=1),
        go.Bar(name='Series', x=years, y=years_s, yaxis='y', offsetgroup=2)
    ],
    layout={
        'yaxis': {'title': 'Count'},
        'xaxis': {'title': 'Years'}
    }
)
fig3.update_layout(
    barmode='group',
    autosize = True,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=5, r=0, t=25, b=0)
)

## Genres bar chart
g_graph = px.bar(
    g,
    color_discrete_sequence=px.colors.sequential.Peach_r,
    labels={0:'Count'},
)
g_graph.update_layout(
    xaxis_title="Genres",
    yaxis_title="Count",
    showlegend=False,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, t=0, b=0),
)    
g_graph.add_annotation(
    x = 'Film-Noir',
    text = 'Only twice',
    showarrow=True,
    arrowhead=1
)

## Map graph
c_graph = px.choropleth(
    data_frame=countries,
    locations='code',
    color='count',
    hover_name='name',
    labels={'count':'Content available'},
    color_continuous_scale=px.colors.sequential.Peach_r,
    height = 550
)
c_graph.update_geos(resolution=50,showocean=True, oceancolor="LightBlue",)
c_graph.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)


app.layout = dbc.Container(children=[
    html.H1("Netflix dataset analysis", className= 'p-6 mt-2'),
    dbc.Row(align = 'center', children = [
        dbc.Tabs(id = 'tabs', active_tab = 'series_movies', children = [
            dbc.Tab(label = 'Series and movies', tab_id = 'series_movies', children = [
                html.H2("Series and Movies distribution", className = 'p-3'),
                html.Div(children = [
                    html.Div(className = 'static_graph', children = [
                        dcc.Graph(figure=pie_1,config={'displayModeBar': False})
                    ]),
                ]),
            ]),
            dbc.Tab(label = 'Content per country', children = [
                html.H2("Content available on each country", className = 'p-3'),
                html.Div(children = [
                    html.Div(className = 'static_graph', children = [
                        dcc.Graph(figure=c_graph,config={'displayModeBar': False})
                    ]),
                ]),
            ]),
            dbc.Tab(label = 'Languages available', children = [
                html.H2("Languages available", className = 'p-3'),
                html.Div(children = [
                    html.Div(className = 'static_graph', children = [
                        dcc.Graph(figure=pie_2,config={'displayModeBar': False})
                    ]),
                ]),
            ]),
            dbc.Tab(label = 'Release date', children = [
                html.H2("Release date of content",className = 'p-3'),
                html.Div(children = [
                    html.Div(className = 'static_graph', children = [
                        dcc.Graph(figure=fig3,config={'displayModeBar': False})
                    ]),
                ]),
            ]),
            dbc.Tab(label = 'Genres', children = [
                html.H2("Genres on Netflix", className = 'p-3'),
                html.Div(children = [
                    html.Div(className = 'static_graph', children = [
                        dcc.Graph(figure=g_graph,config={'displayModeBar': False})
                    ]),
                ]),
            ]),
            dbc.Tab(label = 'Rating by genre', children = [
                dbc.Row(children = [
                    html.H2("Top rated movies by genre", className = 'p-3'),
                    dbc.Col(md = 3, children = [
                        html.Div(children=[
                                html.H4('Select genre', className= 'p-1'),
                                dcc.Dropdown(id="g_slct",
                                                options=[{'label':x,'value':x} for x in genres['index']],
                                                multi=False,
                                                value="Drama"
                                ),
                                html.H4('Select platform', className= 'p-1'),
                                dcc.Dropdown(id="app_slct",
                                                options=[
                                                    {'label':'IMDb','value':'IMDb Score'},
                                                    {'label':'Metacritic','value':'Metacritic Score'},
                                                    {'label':'Rotten Tomatoes','value':'Rotten Tomatoes Score'},
                                                    {'label':'Hidden Gem Score (Netflix)','value':'Hidden Gem Score'}
                                                ],
                                                multi=False,
                                                value="IMDb Score"
                                ),
                                html.H4('Select type', className= 'p-1'),
                                dcc.Dropdown(id="t_slct",
                                                options=[
                                                    {'label':'All','value':'All'},
                                                    {'label':'Movies only','value':'Movie'},
                                                    {'label':'Series only','value':'Series'}
                                                ],
                                                multi=False,
                                                value="All"
                                ),
                        ]),
                    ]),
                    dbc.Col(md = 9, children = [
                        html.Div(children = [
                            dcc.Graph(id='graph1', figure={},config={'displayModeBar': False})
                        ]),
                    ])
                ])
            ]),
            dbc.Tab(label = 'Crew count and average rating', children = [
                dbc.Row(children = [
                    html.H2("Crew members appearances on Netflix and average IMDb rating", className = 'p-3'),
                    dbc.Col(md = 3, children = [
                        dcc.Dropdown(id="c_slct",
                                        options=[
                                            {'label':'Directors','value':'Director'},
                                            {'label':'Actors','value':'Actor'},
                                            {'label':'Writers','value':'Writer'}
                                        ],
                                        multi=False,
                                        value="Director"
                        )
                    ]),
                    dbc.Col(md = 9, children = [
                        html.Div(children = [
                            dcc.Graph(id='graph2', figure={},config={'displayModeBar': False})
                        ]),
                    ])
                ])
            ])
        ])
    ])
])

## Callbacks to update rating graph in real time
@app.callback(
    Output(component_id='graph1', component_property='figure'),
    [Input(component_id='g_slct', component_property='value'),
    Input(component_id='app_slct', component_property='value'),
    Input(component_id='t_slct', component_property='value')]
)

## Function which get genre, platform and type of content to update graph in real time
def update_graph1(g_slct,app_slct,t_slct):

    ## Select the columns to be used
    content = df[['Title','Genre',app_slct,'Series or Movie']]
    content.dropna(inplace=True)
    ## Find the content which has the specified genre
    content = content.loc[content['Genre'].str.contains(g_slct)]
    if t_slct != 'All':
        content = content.loc[content['Series or Movie'] == t_slct]
    ## Sort values by the chosen platform column, and return the top 10
    content.sort_values(by = app_slct,ascending = False,inplace=True)
    t10 = content.head(10)

    range = [0,100]
    if app_slct != 'IMDb Score' and app_slct != 'Hidden Gem Score':
        range = [0,100]
    else:
        range = [0.0,10.0]

    ## Horizontal bar chart with the info from the sorted values
    fig = px.bar(
        t10,
        x=app_slct,
        y = 'Title',
        color_discrete_sequence=px.colors.sequential.Peach_r,
        width=1000,
        height=400,
        orientation='h',
    )
    fig.update_xaxes(range=range)
    fig.update_layout(
        xaxis_title="Rating",
        yaxis_title="Titles",
        autosize = True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=5, r=0, t=25, b=0),
    )

    return fig

## Callback to update crew members graph in real time
@app.callback(
    Output(component_id='graph2', component_property='figure'),
    Input(component_id='c_slct', component_property='value')
)

def update_graph2(c_slct):
    ## Three empty arrays, that later hold the info to be shown
    count = []
    names = []
    average = []
    ## Fill the arrays with the info requested through the select button
    if c_slct == 'Director':
        count = directors['count']
        names = directors['director']
        average = directors['avg']
    elif c_slct == 'Actor':
        count = actors['count']
        names = actors['actor']
        average = actors['avg']
    elif c_slct == 'Writer':
        count = writers['count']
        names = writers['writer']
        average = writers['avg']
    
    ## Bar chart with two y axis
    fig2 = go.Figure(
        data = [
            go.Bar(name='Count', x=names, y=count, yaxis='y', offsetgroup=1),
            go.Bar(name='Average Rating', x=names, y=average, yaxis='y2', offsetgroup=2)
        ],
        layout={
            'yaxis': {'title': 'Count'},
            'yaxis2': {'title': 'Average Rating', 'overlaying': 'y', 'side': 'right'},
            'xaxis': {'title': str(c_slct)}
        }
    )
    fig2.update_layout(
        barmode='group',
        autosize = True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=5, r=0, t=25, b=0)
    )

    return fig2
    
if __name__ == '__main__':
    app.run_server(debug=True)