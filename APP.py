import streamlit as st
import pandas as pd
import Final_Model as fn
import base64
import os


st.title("Teste da Qualidade do Recozimento")

st.sidebar.markdown('Para utilizar o aplicativo, utilize o botão abaixo para ' + 
                    'fazer o upload da sua base de dados que será passada ' + 
                    'ao algoritmo para realizar a predição da qualidade do ' + 
                    'recozimento de suas peças metálicas.')
                    

file = st.sidebar.file_uploader("# Faça o upload dos dados abaixo", type=['csv'])

@st.cache(allow_output_mutation=True)
def button_states():
    return {"pressed": False}

st.sidebar.markdown('Caso ainda não tenha um arquivo formatado para inserir ' + 
                    'os dados no algoritmo de predição, clique no link ' + 
                    'abaixo para fazer o download: ')


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

st.sidebar.markdown(get_binary_file_downloader_html('test_format.csv', 'Formato Padrão'), unsafe_allow_html=True)


if file is not None:
    df = pd.read_csv(file)
    st.dataframe(df)
    st.markdown('Os dados carregados estão de acordo? Em caso afirmativo, ' + 
                'clicar no botão "Run Predictions".')
    run_button = st.button('Run Prediction')
    run_state = button_states()
    
    if run_button:
        run_state.update({'pressed': True})
        
    if run_state['pressed']:
        st.markdown('Previsão do recozimento feita pelo algoritmo:')
        outputs = fn.final_model(df)
        col1, col2 = st.columns(2)
        col1.dataframe(outputs)
        col2.markdown('Caso queira exportar a previsão realizada, clicar no ' +
                      'botão abaixo.')
        export_button = col2.button('Export Data')
        
        if export_button:
            run_button = True
            csv = outputs.to_csv(index = False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings
            linko = f'<a href="data:file/csv;base64,{b64}" download="Predictions.csv">Download csv file</a>'
            col2.markdown(linko, unsafe_allow_html=True)
