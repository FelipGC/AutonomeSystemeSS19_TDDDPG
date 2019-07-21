import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import log_parser as parser
import params as params


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

n_clicks = None
app = dash.Dash(__name__)

PARAMETERS_TO_CHANGE = {
    'POLICY_DELAY': ['1', '2', '4', '8', '16'],
    'TAU': ['1', '0.1', '0.01', '0.001', '0.0001', '0.00001'],
    'BUFFER_SIZE': ['8192', '262144', '1048576'],
    'UNITS': ['64', '128', '12812864'],
    'BATCH_SIZE': ['32', '64'],
    'INITIAL_RANDOM_ROLLOUTS': ['True', 'False'],
    'DROP_OUT_PROB': ['0', '0.1', '0.2'],
}


app.layout = html.Div([
html.Label('Dashboard', id="title_dashboard"),
    html.Div([
        html.Div([

            dcc.Dropdown(
                id='dropdown_hyperparams_1',
                options=[{'label': i, 'value': i} for i in PARAMETERS_TO_CHANGE.keys()],
                value='POLICY_DELAY',
                style={'width': '100%'}
            ),

            dcc.Dropdown(
                id='dropdown_twin_delay_1',
                options=[{'label': i, 'value': i} for i in ['True', 'False']],
                value='False',
                style={'width': '100%'}
            )], style={'display': 'flex'}),

        html.Div([
            html.Div([
                html.Label('Values Hyperparameter: ',id='textInput_hyperparam_1'),
                dcc.RadioItems(id='radio_hyperparams_1', labelStyle={'display': 'inline-block'})
            ], className='rb_layout_1'),

            html.Button('Enlarge', id='button_autoscale_1',value="", type="reset")

        ], className='center', style={'display': 'flex'}),


        html.Div(id='container-button-basic',
                     children='Enter a value and press submit'),

        dcc.Graph(id='scatter_plot_1'),
        html.Div(id='textOutput_sum_1')

    ], className='half'),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='dropdown_hyperparams_2',
                options=[{'label': i, 'value': i} for i in PARAMETERS_TO_CHANGE.keys()],
                value='BUFFER_SIZE',
                style={'width': '100%'}
            ),

            dcc.Dropdown(
                id='dropdown_twin_delay_2',
                options=[{'label': i, 'value': i} for i in ['True', 'False']],
                value='False',
                style={'width': '100%'}

            )], style={'display': 'flex'}),
        html.Div([
            html.Div([
                html.Label('Values Hyperparameter: ',id='textInput_hyperparam_2'),
                dcc.RadioItems(id='radio_hyperparams_2', labelStyle={'display': 'inline-block'})
            ], className='rb_layout_1'),

            html.Button('Enlarge', id='button_autoscale_2',value="", type="reset")
        ], className='center'),

        html.Div(id='container-button-basic_2',
                             children=''),

        dcc.Graph(id='scatter_plot_2'),
        html.Div(id='textOutput_sum_2')

    ], className='half'),
    html.Div([
            html.Div([
                dcc.Dropdown(
                    id='dropdown_hyperparams_5',
                    options=[{'label': i, 'value': i} for i in PARAMETERS_TO_CHANGE.keys()],
                    value='POLICY_DELAY',
                    style={'width': '100%'}
                ),

                dcc.Dropdown(
                    id='dropdown_twin_delay_5',
                    options=[{'label': i, 'value': i} for i in ['Min', 'Max', 'Median','Average','Current']],
                    value='Average',
                    style={'width': '100%'}
                )], style={'display': 'flex'}),
            html.Button('Enlarge', id='button_autoscale_5',value="", type="reset"),
            html.Div(id='container-button-basic_5',
                                         children=''),

            dcc.Graph(id='scatter_plot_5'),

        ], className='half'),
    html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='dropdown_hyperparams_6',
                        options=[{'label': i, 'value': i} for i in PARAMETERS_TO_CHANGE.keys()],
                        value='TAU',
                        style={'width': '100%'}
                    ),

                    dcc.Dropdown(
                        id='dropdown_twin_delay_6',
                        options=[{'label': i, 'value': i} for i in ['Min', 'Max', 'Median','Average','Current']],
                        value='Average',
                        style={'width': '100%'}
                    )], style={'display': 'flex'}),
                html.Button('Enlarge', id='button_autoscale_6',value="", type="reset"),

                html.Div(id='container-button-basic_6',
                                             children=''),


                dcc.Graph(id='scatter_plot_6'),

            ], className='half')
])

layout = dict(title='Linechart',
              xaxis=dict(title='Episodes'),
              yaxis=dict(title='Score'),
              )

################### Values Autosize #####################

selected_hyperparam_1, selected_value_1, twin_delay_1 = None, None, None
selected_hyperparam_2, selected_value_2, twin_delay_2 = None, None, None
selected_hyperparam_5, selected_evaluation_5 = None, None
selected_hyperparam_6, selected_evaluation_6 = None, None

########################## 1 ############################



@app.callback(
    Output('scatter_plot_1', 'figure'),
    [Input('dropdown_hyperparams_1', 'value'),
     Input('radio_hyperparams_1', 'value'),
     Input('dropdown_twin_delay_1', 'value')])
def update_graph(selected_hyperparam, selected_value, twin_delay):
    global selected_hyperparam_1, selected_value_1, twin_delay_1, n_clicks
    selected_hyperparam_1 = selected_hyperparam
    selected_value_1 = selected_value
    twin_delay_1 = twin_delay
    try:
        figure = parser.get_avg_graphs(selected_hyperparam, selected_value, twin_delay)
        n_clicks = None
        return figure[0]
    except:
        print("In the directory '" + params.PATH_DASH + "' are no logs for the selected hyperparameters (Plot 1)")


@app.callback(
    Output('textOutput_sum_1', 'children'),
    [Input('dropdown_hyperparams_1', 'value'),
     Input('radio_hyperparams_1', 'value'),
     Input('dropdown_twin_delay_1', 'value'),
     Input('scatter_plot_1', 'figure')])
def set_radio_value(selected_hyperparam, selected_value, twin_delay, figure):
    sum_1 = parser.sum_rewards
    return u'Sum Rewards: {}'.format(sum_1)


@app.callback(
    Output('radio_hyperparams_1', 'options'),
    [Input('dropdown_hyperparams_1', 'value')])
def set_hyperparam_options(selected_hyperparam):
    return [{'label': i, 'value': i} for i in PARAMETERS_TO_CHANGE[selected_hyperparam]]


@app.callback(
    Output('radio_hyperparams_1', 'value'),
    [Input('radio_hyperparams_1', 'options')])
def set_radio_value(available_options):
    return available_options[0]['value']


########################## 2 ############################

@app.callback(
    Output('scatter_plot_2', 'figure'),
    [Input('dropdown_hyperparams_2', 'value'),
     Input('radio_hyperparams_2', 'value'),
     Input('dropdown_twin_delay_2', 'value')])
def update_graph(selected_hyperparam, selected_value, twin_delay):
    global selected_hyperparam_2, selected_value_2, twin_delay_2
    selected_hyperparam_2 = selected_hyperparam
    selected_value_2 = selected_value
    twin_delay_2 = twin_delay
    try:
        figure = parser.get_avg_graphs(selected_hyperparam, selected_value, twin_delay)
        return figure[0]
    except:
        print("In the directory '" + params.PATH_DASH + "' are no logs for the selected hyperparameters (Plot 2)")


@app.callback(
    Output('radio_hyperparams_2', 'options'),
    [Input('dropdown_hyperparams_2', 'value')])
def set_hyperparam_options(selected_hyperparam):
    return [{'label': i, 'value': i} for i in PARAMETERS_TO_CHANGE[selected_hyperparam]]


@app.callback(
    Output('radio_hyperparams_2', 'value'),
    [Input('radio_hyperparams_2', 'options')])
def set_radio_value(available_options):
    return available_options[0]['value']

@app.callback(
    Output('textOutput_sum_2', 'children'),
    [Input('dropdown_hyperparams_2', 'value'),
     Input('radio_hyperparams_2', 'value'),
     Input('dropdown_twin_delay_2', 'value'),
     Input('scatter_plot_2', 'figure')])
def set_radio_value(selected_hyperparam, selected_value, twin_delay, figure):
    sum_2 = parser.sum_rewards
    return u'Sum Rewards: {}'.format(sum_2)


########################## 5 ############################

@app.callback(
    Output('scatter_plot_5', 'figure'),
    [Input('dropdown_hyperparams_5', 'value'),
     Input('dropdown_twin_delay_5', 'value')])
def update_graph(selected_hyperparam, selected_evaluation):
    global selected_hyperparam_5, selected_evaluation_5
    selected_hyperparam_5 = selected_hyperparam
    selected_evaluation_5 = selected_evaluation
    try:
        figure = parser.get_multi_value_graphs(selected_hyperparam, selected_evaluation)
        return figure[0]
    except:
        print("In the directory '" + params.PATH_DASH + "' are no logs for the selected hyperparameters (Plot 3)")


########################## 6 ############################

@app.callback(
    Output('scatter_plot_6', 'figure'),
    [Input('dropdown_hyperparams_6', 'value'),
     Input('dropdown_twin_delay_6', 'value')])
def update_graph(selected_hyperparam, selected_evaluation):
    global selected_hyperparam_6, selected_evaluation_6
    selected_hyperparam_6 = selected_hyperparam
    selected_evaluation_6 = selected_evaluation
    try:
        figure = parser.get_multi_value_graphs(selected_hyperparam, selected_evaluation)
        return figure[0]
    except:
        print("In the directory '" + params.PATH_DASH + "' are no logs for the selected hyperparameters (Plot 4)")


###################### Button 1 ########################

@app.callback(
    dash.dependencies.Output('container-button-basic', 'children'),
    [dash.dependencies.Input('button_autoscale_1', 'n_clicks')])
def update_output(n_clicks):
    if n_clicks is not None:
        parser.display_graphs(parser.get_avg_graphs(selected_hyperparam_1, selected_value_1, twin_delay_1), selected_hyperparam_1, twin_delay_1)


###################### Button 2 ########################

@app.callback(
    dash.dependencies.Output('container-button-basic_2', 'children'),
    [dash.dependencies.Input('button_autoscale_2', 'n_clicks')])
def update_output(n_clicks):
    if n_clicks is not None:
        parser.display_graphs(parser.get_avg_graphs(selected_hyperparam_2, selected_value_2, twin_delay_2), selected_hyperparam_2, twin_delay_2)


###################### Button 3 ########################

@app.callback(
    dash.dependencies.Output('container-button-basic_5', 'children'),
    [dash.dependencies.Input('button_autoscale_5', 'n_clicks')])
def update_output(n_clicks):
    if n_clicks is not None:
        parser.display_graphs(parser.get_multi_value_graphs(selected_hyperparam_5, selected_evaluation_5), selected_hyperparam_5, selected_evaluation_5)


###################### Button 4 ########################

@app.callback(
    dash.dependencies.Output('container-button-basic_6', 'children'),
    [dash.dependencies.Input('button_autoscale_6', 'n_clicks')])
def update_output(n_clicks):
    if n_clicks is not None:
        parser.display_graphs(parser.get_multi_value_graphs(selected_hyperparam_6, selected_evaluation_6), selected_hyperparam_6, selected_evaluation_6)


if __name__ == '__main__':
    app.run_server(debug=True)
