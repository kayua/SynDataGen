#!/usr/bin/env python3
"""
Servidor Flask Unificado
- Monitoramento de Sistema em Tempo Real
- Gerenciamento de Experimentos SynDataGen
- Execução automática do main.py ao receber experimentos (DAEMON)
- Recuperação de resultados dos experimentos
"""

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import psutil
import platform
import time
from datetime import datetime
import json
import os
import subprocess
import threading
import shlex
import signal

app = Flask(__name__)
CORS(app)

# ==================== CONFIGURAÇÕES ====================

# Histórico de monitoramento
cpu_history = []
memory_history = []
MAX_HISTORY = 60

# Diretório para salvar experimentos
EXPERIMENTS_DIR = 'outputs'
if not os.path.exists(EXPERIMENTS_DIR):
    os.makedirs(EXPERIMENTS_DIR)

# Diretório de outputs dos experimentos
OUTPUTS_DIR = 'outputs'

experiments_list = []

# Dicionário para rastrear processos em execução
running_processes = {}

# ==================== GERENCIAMENTO DE DATASETS ====================

# Diretório para datasets
DATASETS_DIR = 'Dataset'
if not os.path.exists(DATASETS_DIR):
    os.makedirs(DATASETS_DIR)


@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """Lista todos os datasets disponíveis com informações detalhadas"""
    try:
        datasets = []

        if not os.path.exists(DATASETS_DIR):
            return jsonify({
                'status': 'success',
                'count': 0,
                'datasets': []
            }), 200

        # Itera sobre todos os arquivos no diretório de datasets
        for filename in os.listdir(DATASETS_DIR):
            filepath = os.path.join(DATASETS_DIR, filename)

            # Ignora diretórios
            if not os.path.isfile(filepath):
                continue

            # Obtém informações do arquivo
            file_stats = os.stat(filepath)
            file_size = file_stats.st_size
            created_at = datetime.fromtimestamp(file_stats.st_ctime).isoformat()
            modified_at = datetime.fromtimestamp(file_stats.st_mtime).isoformat()

            # Determina o tipo de arquivo
            file_extension = os.path.splitext(filename)[1].lower()

            dataset_info = {
                'filename': filename,
                'filepath': filepath,
                'size_bytes': file_size,
                'size_mb': round(file_size / (1024 ** 2), 2),
                'extension': file_extension,
                'created_at': created_at,
                'modified_at': modified_at
            }

            # Adiciona informações específicas para CSVs
            if file_extension == '.csv':
                try:
                    import pandas as pd
                    df = pd.read_csv(filepath, nrows=5)  # Lê apenas primeiras 5 linhas
                    dataset_info.update({
                        'rows': len(pd.read_csv(filepath)),
                        'columns': len(df.columns),
                        'column_names': df.columns.tolist(),
                        'preview': df.head().to_dict('records')
                    })
                except Exception as e:
                    dataset_info['error'] = f'Erro ao ler CSV: {str(e)}'

            datasets.append(dataset_info)

        # Ordena por data de modificação (mais recentes primeiro)
        datasets.sort(key=lambda x: x.get('modified_at', ''), reverse=True)

        return jsonify({
            'status': 'success',
            'count': len(datasets),
            'datasets': datasets,
            'directory': os.path.abspath(DATASETS_DIR)
        }), 200

    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': f'Erro ao listar datasets: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/datasets/<path:filename>', methods=['GET'])
def get_dataset_info(filename):
    """Retorna informações detalhadas de um dataset específico"""
    try:
        filepath = os.path.join(DATASETS_DIR, filename)

        if not os.path.exists(filepath):
            return jsonify({
                'status': 'error',
                'message': 'Dataset não encontrado'
            }), 404

        if not os.path.isfile(filepath):
            return jsonify({
                'status': 'error',
                'message': 'O caminho especificado não é um arquivo'
            }), 400

        # Obtém informações do arquivo
        file_stats = os.stat(filepath)
        file_size = file_stats.st_size
        created_at = datetime.fromtimestamp(file_stats.st_ctime).isoformat()
        modified_at = datetime.fromtimestamp(file_stats.st_mtime).isoformat()

        file_extension = os.path.splitext(filename)[1].lower()

        dataset_info = {
            'filename': filename,
            'filepath': filepath,
            'size_bytes': file_size,
            'size_mb': round(file_size / (1024 ** 2), 2),
            'extension': file_extension,
            'created_at': created_at,
            'modified_at': modified_at
        }

        # Informações adicionais para CSVs
        if file_extension == '.csv':
            try:
                import pandas as pd
                df = pd.read_csv(filepath)
                dataset_info.update({
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': df.columns.tolist(),
                    'dtypes': df.dtypes.astype(str).to_dict(),
                    'preview': df.head(10).to_dict('records'),
                    'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2)
                })
            except Exception as e:
                dataset_info['error'] = f'Erro ao ler CSV: {str(e)}'

        return jsonify({
            'status': 'success',
            'dataset': dataset_info
        }), 200

    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': f'Erro ao obter informações do dataset: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/datasets/upload', methods=['POST'])
def upload_dataset():
    """Upload de um novo dataset com verificação de colisão de nomes"""
    try:
        # Verifica se há arquivo no request
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Nenhum arquivo foi enviado'
            }), 400

        file = request.files['file']

        # Verifica se o arquivo tem nome
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'Nenhum arquivo selecionado'
            }), 400

        # Obtém o nome original do arquivo
        original_filename = file.filename
        base_name, extension = os.path.splitext(original_filename)

        # Verifica se deve sobrescrever (parâmetro opcional)
        overwrite = request.form.get('overwrite', 'false').lower() == 'true'

        # Gera um nome único se necessário
        final_filename = original_filename
        counter = 1

        if not overwrite:
            while os.path.exists(os.path.join(DATASETS_DIR, final_filename)):
                final_filename = f"{base_name}_{counter}{extension}"
                counter += 1

        # Salva o arquivo
        filepath = os.path.join(DATASETS_DIR, final_filename)
        file.save(filepath)

        # Obtém informações do arquivo salvo
        file_stats = os.stat(filepath)
        file_size = file_stats.st_size

        response_data = {
            'status': 'success',
            'message': 'Dataset enviado com sucesso',
            'dataset': {
                'original_filename': original_filename,
                'saved_filename': final_filename,
                'filepath': filepath,
                'size_bytes': file_size,
                'size_mb': round(file_size / (1024 ** 2), 2),
                'created_at': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                'was_renamed': final_filename != original_filename
            }
        }

        # Se for CSV, adiciona informações extras
        if extension.lower() == '.csv':
            try:
                import pandas as pd
                df = pd.read_csv(filepath)
                response_data['dataset'].update({
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': df.columns.tolist()
                })
            except Exception as e:
                response_data['dataset']['csv_info_error'] = str(e)

        print(f"\n✅ Dataset enviado: {final_filename}")
        print(f"   Tamanho: {response_data['dataset']['size_mb']} MB")
        print(f"   Localização: {filepath}\n")

        return jsonify(response_data), 201

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()

        print(f"\n❌ Erro ao fazer upload do dataset!")
        print(f"Erro: {str(e)}")
        print(f"Detalhes: {error_details}\n")

        return jsonify({
            'status': 'error',
            'message': f'Erro ao fazer upload do dataset: {str(e)}',
            'traceback': error_details
        }), 500


@app.route('/api/datasets/download/<path:filename>', methods=['GET'])
def download_dataset(filename):
    """Download de um dataset específico"""
    try:
        filepath = os.path.join(DATASETS_DIR, filename)

        if not os.path.exists(filepath):
            return jsonify({
                'status': 'error',
                'message': 'Dataset não encontrado'
            }), 404

        if not os.path.isfile(filepath):
            return jsonify({
                'status': 'error',
                'message': 'O caminho especificado não é um arquivo'
            }), 400

        print(f"\n📥 Download solicitado: {filename}")

        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': f'Erro ao fazer download do dataset: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/datasets/<path:filename>', methods=['DELETE'])
def delete_dataset(filename):
    """Deleta um dataset específico"""
    try:
        filepath = os.path.join(DATASETS_DIR, filename)

        if not os.path.exists(filepath):
            return jsonify({
                'status': 'error',
                'message': 'Dataset não encontrado'
            }), 404

        if not os.path.isfile(filepath):
            return jsonify({
                'status': 'error',
                'message': 'O caminho especificado não é um arquivo'
            }), 400

        # Verifica se o dataset está sendo usado por algum experimento em execução
        dataset_in_use = False
        experiments_using = []

        for exp_id, proc_info in running_processes.items():
            if proc_info.get('status') in ['starting', 'running']:
                cmd = proc_info.get('command', '')
                if filename in cmd or filepath in cmd:
                    dataset_in_use = True
                    experiments_using.append(exp_id)

        if dataset_in_use:
            return jsonify({
                'status': 'error',
                'message': 'Dataset está sendo usado por experimentos em execução',
                'experiments': experiments_using
            }), 409  # Conflict

        # Remove o arquivo
        os.remove(filepath)

        print(f"\n🗑️  Dataset removido: {filename}\n")

        return jsonify({
            'status': 'success',
            'message': 'Dataset removido com sucesso',
            'filename': filename
        }), 200

    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': f'Erro ao deletar dataset: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


# ==================== FUNÇÕES AUXILIARES ====================

def run_cmd_daemon(cmd, experiment_id):
    """
    Executa um comando shell em background como daemon.
    O processo é completamente desacoplado do servidor Flask.
    Multiplataforma (Windows/Linux/Mac).
    """
    try:
        print(f"\n{'='*80}")
        print(f"🚀 Executando comando daemon:")
        print(f"📋 {cmd}")
        print(f"{'='*80}\n")

        args = shlex.split(cmd)

        # Cria diretório de logs para o experimento
        log_dir = os.path.join(OUTPUTS_DIR, f"exp_{experiment_id}", "logs")
        os.makedirs(log_dir, exist_ok=True)

        stdout_log = os.path.join(log_dir, "stdout.log")
        stderr_log = os.path.join(log_dir, "stderr.log")

        print(f"📁 Diretório de logs: {log_dir}")
        print(f"📄 STDOUT: {stdout_log}")
        print(f"📄 STDERR: {stderr_log}\n")

        # Abre arquivos de log
        stdout_file = open(stdout_log, 'w', buffering=1)  # Line buffering
        stderr_file = open(stderr_log, 'w', buffering=1)

        # Configurações específicas por plataforma
        popen_kwargs = {
            'args': args,
            'stdout': stdout_file,
            'stderr': stderr_file,
            'stdin': subprocess.DEVNULL,
            'cwd': os.getcwd()
        }

        # Adiciona configurações de daemon por plataforma
        if platform.system() != 'Windows':
            # Em sistemas Unix, usa apenas start_new_session para desacoplar
            # Isso é suficiente para criar um processo daemon
            popen_kwargs['start_new_session'] = True
            print("🐧 Modo Unix: processo será desacoplado (daemon)")
        else:
            # Windows: usa CREATE_NEW_PROCESS_GROUP
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            DETACHED_PROCESS = 0x00000008
            popen_kwargs['creationflags'] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            print("🪟 Modo Windows: processo em background")

        print(f"⚙️  Iniciando processo...\n")

        # Inicia o processo em background (daemon)
        process = subprocess.Popen(**popen_kwargs)

        # Registra o PID e informações do processo
        process_info = {
            'pid': process.pid,
            'stdout_log': stdout_log,
            'stderr_log': stderr_log,
            'stdout_file': stdout_file,
            'stderr_file': stderr_file,
            'process': process
        }

        print(f"✅ Processo daemon iniciado com sucesso!")
        print(f"🆔 PID: {process.pid}")
        print(f"📊 Status: RUNNING")
        print(f"{'='*80}\n")

        return True, process_info

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()

        print(f"\n{'='*80}")
        print(f"❌ ERRO ao executar comando daemon!")
        print(f"{'='*80}")
        print(f"Erro: {str(e)}")
        print(f"\nDetalhes completos:")
        print(error_details)
        print(f"{'='*80}\n")

        return False, str(e)


def monitor_process_daemon(experiment_id, process_info):
    """
    Monitora o processo daemon em background até sua conclusão.
    Atualiza o status do experimento conforme o processo executa.
    """
    try:
        process = process_info['process']
        pid = process_info['pid']

        print(f"\n🔍 Iniciando monitoramento do experimento {experiment_id} (PID: {pid})")

        # Aguarda o processo terminar (sem bloquear o servidor)
        return_code = process.wait()

        # Fecha os arquivos de log
        try:
            process_info['stdout_file'].flush()
            process_info['stderr_file'].flush()
            process_info['stdout_file'].close()
            process_info['stderr_file'].close()
        except:
            pass

        # Atualiza o status final
        end_time = datetime.now().isoformat()

        print(f"\n{'='*80}")
        if return_code == 0:
            running_processes[experiment_id]['status'] = 'completed'
            print(f"✅ Experimento {experiment_id} CONCLUÍDO COM SUCESSO!")
            print(f"🆔 PID: {pid}")
            print(f"⏱️  Fim: {end_time}")
            print(f"🎯 Return Code: {return_code}")
        else:
            running_processes[experiment_id]['status'] = 'failed'
            running_processes[experiment_id]['return_code'] = return_code
            print(f"❌ Experimento {experiment_id} FALHOU!")
            print(f"🆔 PID: {pid}")
            print(f"⏱️  Fim: {end_time}")
            print(f"⚠️  Return Code: {return_code}")

        running_processes[experiment_id]['end_time'] = end_time

        # Lê os últimos 1000 caracteres do stderr para diagnóstico
        try:
            stderr_log = process_info['stderr_log']
            if os.path.exists(stderr_log) and os.path.getsize(stderr_log) > 0:
                with open(stderr_log, 'r') as f:
                    stderr_content = f.read()
                    if stderr_content:
                        running_processes[experiment_id]['stderr_preview'] = stderr_content[-1000:]
                        print(f"⚠️  Stderr disponível: {len(stderr_content)} caracteres")
                        if return_code != 0:
                            print(f"\n📋 Últimas linhas do stderr:")
                            print("-" * 80)
                            print(stderr_content[-500:])
                            print("-" * 80)
        except Exception as e:
            print(f"⚠️  Erro ao ler stderr: {e}")

        print(f"📊 Status final: {running_processes[experiment_id]['status']}")
        print(f"📁 Logs: {process_info['stdout_log']}")
        print(f"{'='*80}\n")

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()

        print(f"\n{'='*80}")
        print(f"❌ ERRO ao monitorar experimento {experiment_id}!")
        print(f"{'='*80}")
        print(f"Erro: {str(e)}")
        print(f"\nDetalhes:")
        print(error_details)
        print(f"{'='*80}\n")

        running_processes[experiment_id]['status'] = 'error'
        running_processes[experiment_id]['error'] = str(e)
        running_processes[experiment_id]['error_details'] = error_details
        running_processes[experiment_id]['end_time'] = datetime.now().isoformat()


def run_experiment_async(experiment_id, cmd):
    """
    Inicia o experimento como daemon e configura o monitoramento.
    """
    try:
        print(f"\n{'='*80}")
        print(f"🧪 NOVO EXPERIMENTO RECEBIDO")
        print(f"{'='*80}")
        print(f"🆔 ID: {experiment_id}")
        print(f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")

        # Registra o processo como em execução
        running_processes[experiment_id] = {
            'status': 'starting',
            'start_time': datetime.now().isoformat(),
            'command': cmd
        }

        # Executa o comando como daemon
        success, result = run_cmd_daemon(cmd, experiment_id)

        if not success:
            running_processes[experiment_id]['status'] = 'failed'
            running_processes[experiment_id]['error'] = result
            running_processes[experiment_id]['end_time'] = datetime.now().isoformat()

            print(f"\n{'='*80}")
            print(f"❌ FALHA ao iniciar experimento {experiment_id}!")
            print(f"{'='*80}")
            print(f"Erro: {result}")
            print(f"{'='*80}\n")
            return

        # Atualiza informações do processo daemon
        running_processes[experiment_id].update({
            'status': 'running',
            'pid': result['pid'],
            'stdout_log': result['stdout_log'],
            'stderr_log': result['stderr_log']
        })

        # Inicia thread de monitoramento (não bloqueia)
        monitor_thread = threading.Thread(
            target=monitor_process_daemon,
            args=(experiment_id, result),
            daemon=True,
            name=f"Monitor-{experiment_id}"
        )
        monitor_thread.start()

        print(f"🔍 Thread de monitoramento iniciada: {monitor_thread.name}\n")

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()

        print(f"\n{'='*80}")
        print(f"❌ EXCEÇÃO ao iniciar experimento {experiment_id}!")
        print(f"{'='*80}")
        print(f"Erro: {str(e)}")
        print(f"\nStack trace:")
        print(error_details)
        print(f"{'='*80}\n")

        running_processes[experiment_id]['status'] = 'error'
        running_processes[experiment_id]['error'] = str(e)
        running_processes[experiment_id]['error_details'] = error_details
        running_processes[experiment_id]['end_time'] = datetime.now().isoformat()


def build_main_command(experiment_data):
    """
    Constrói o comando para executar o main.py com base nos dados do experimento
    Formato similar ao usado em campaigns.py
    """
    cmd = "python3 main.py"

    # Campos de metadados que não devem ser passados como argumentos
    # Note: dataset_path não está aqui pois deve ser tratado especialmente
    metadata_fields = {'experiment_name', 'id', 'timestamp', 'received_at'}

    # Adiciona o caminho do dataset
    # Verifica primeiro no nível superior, depois em configuration
    dataset_path = None
    if 'dataset_path' in experiment_data and experiment_data['dataset_path']:
        dataset_path = experiment_data['dataset_path']
    elif 'configuration' in experiment_data and 'dataset_path' in experiment_data['configuration']:
        dataset_path = experiment_data['configuration']['dataset_path']

    if dataset_path:
        cmd += " --data_load_path_file_input {}".format(dataset_path)

    # Adiciona o diretório de saída
    output_dir = "outputs/exp_{}".format(experiment_data['id'])
    cmd += " --output_dir {}".format(output_dir)

    # Adiciona configurações do experimento
    if 'configuration' in experiment_data:
        config = experiment_data['configuration']

        # Itera sobre todas as configurações e adiciona ao comando
        # Formato: --parametro valor (similar ao campaigns.py)
        for param in config.keys():
            # Ignora campos de metadados e dataset_path (já foi adicionado)
            if param in metadata_fields or param == 'dataset_path':
                continue

            value = config[param]
            if value is not None and value != '':
                # Converte listas em strings separadas por espaço
                if isinstance(value, list):
                    value = ' '.join(map(str, value))
                cmd += " --{} {}".format(param, value)

    return cmd


def get_experiment_results(experiment_id):
    """
    Busca os resultados de um experimento específico
    Retorna o conteúdo do Results.json se existir
    """
    results_path = os.path.join(OUTPUTS_DIR, f"exp_{experiment_id}", "EvaluationResults", "Results.json")

    if os.path.exists(results_path):
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Erro ao ler Results.json para {experiment_id}: {str(e)}")
            return None
    return None


def get_experiment_monitors(experiment_id):
    """
    Busca os arquivos de monitor de um experimento específico
    Retorna lista com conteúdo dos monitor_model_*_fold.json
    """
    monitors = []
    monitor_dir = os.path.join(OUTPUTS_DIR, f"exp_{experiment_id}", "Monitor")

    if os.path.exists(monitor_dir):
        try:
            monitor_files = sorted([f for f in os.listdir(monitor_dir) if f.endswith('.json')])
            for monitor_file in monitor_files:
                monitor_path = os.path.join(monitor_dir, monitor_file)
                with open(monitor_path, 'r', encoding='utf-8') as f:
                    monitor_data = json.load(f)
                    monitors.append({
                        'filename': monitor_file,
                        'data': monitor_data
                    })
        except Exception as e:
            print(f"Erro ao ler monitors para {experiment_id}: {str(e)}")

    return monitors


def check_process_status(experiment_id):
    """
    Verifica se um processo ainda está rodando através do PID.
    """
    if experiment_id not in running_processes:
        return None

    proc_info = running_processes[experiment_id]

    # Se já tem status final, retorna
    if proc_info.get('status') in ['completed', 'failed', 'error', 'cancelled']:
        return proc_info.get('status')

    # Verifica se o processo ainda existe
    pid = proc_info.get('pid')
    if pid:
        try:
            # Verifica se o processo existe (não envia sinal, apenas verifica)
            os.kill(pid, 0)
            return 'running'
        except OSError:
            # Processo não existe mais
            proc_info['status'] = 'completed'
            proc_info['end_time'] = datetime.now().isoformat()
            return 'completed'

    return proc_info.get('status', 'unknown')


# ==================== ROTAS PRINCIPAIS ====================

@app.route('/')
def index():
    """Página inicial - Dashboard de Monitoramento"""
    try:
        return send_file('codigo_editado.html')
    except FileNotFoundError:
        return jsonify({
            'message': 'Servidor Unificado - Monitoramento e Experimentos',
            'version': '2.0 (Daemon)',
            'services': {
                'monitoring': 'Sistema de Monitoramento em Tempo Real',
                'experiments': 'Gerenciamento de Experimentos SynDataGen (Background Daemon)'
            },
            'endpoints': {
                'system': {
                    '/': 'GET - Dashboard de monitoramento (HTML)',
                    '/api/system-info': 'GET - Informações do sistema em tempo real'
                },
                'experiments': {
                    '/api/experiments': 'GET - Listar todos experimentos',
                    '/api/experiments': 'POST - Criar novo experimento (executa main.py como daemon)',
                    '/api/experiments/<id>': 'GET - Obter experimento específico',
                    '/api/experiments/<id>': 'DELETE - Deletar experimento',
                    '/api/experiments/<id>/cancel': 'POST - Cancelar experimento em execução',
                    '/api/experiments/<id>/status': 'GET - Status de execução do experimento',
                    '/api/experiments/<id>/logs': 'GET - Logs do experimento',
                    '/api/experiments/results': 'GET - Obter resultados de todos os experimentos',
                    '/api/experiments/<id>/results': 'GET - Obter resultados de um experimento específico'
                }
            }
        })


# ==================== MONITORAMENTO DE SISTEMA ====================

@app.route('/api/system-info', methods=['GET'])
def get_system_info():
    """Retorna informações em tempo real do sistema"""

    # CPU
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
    cpu_freq = psutil.cpu_freq()
    cpu_count = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)

    # Memória
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()

    # Disco
    disk = psutil.disk_usage('/')
    disk_io = psutil.disk_io_counters()

    # Rede
    net_io = psutil.net_io_counters()

    # GPU (tentar obter se disponível)
    gpu_info = get_gpu_info()

    # Processos
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            pinfo = proc.info
            if pinfo['cpu_percent'] is not None and pinfo['cpu_percent'] > 0:
                processes.append(pinfo)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Ordenar por uso de CPU
    processes = sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:10]

    # Sistema
    boot_time = datetime.fromtimestamp(psutil.boot_time()).strftime("%Y-%m-%d %H:%M:%S")
    uptime_seconds = time.time() - psutil.boot_time()

    # Adicionar ao histórico
    cpu_history.append(cpu_percent)
    memory_history.append(memory.percent)

    if len(cpu_history) > MAX_HISTORY:
        cpu_history.pop(0)
    if len(memory_history) > MAX_HISTORY:
        memory_history.pop(0)

    data = {
        'timestamp': datetime.now().isoformat(),
        'cpu': {
            'percent': round(cpu_percent, 2),
            'per_core': [round(x, 2) for x in cpu_per_core],
            'count': cpu_count,
            'physical_count': cpu_count_physical,
            'freq_current': round(cpu_freq.current, 2) if cpu_freq else None,
            'freq_min': round(cpu_freq.min, 2) if cpu_freq else None,
            'freq_max': round(cpu_freq.max, 2) if cpu_freq else None,
            'history': cpu_history.copy()
        },
        'memory': {
            'total': round(memory.total / (1024**3), 2),  # GB
            'available': round(memory.available / (1024**3), 2),
            'used': round(memory.used / (1024**3), 2),
            'percent': round(memory.percent, 2),
            'history': memory_history.copy()
        },
        'swap': {
            'total': round(swap.total / (1024**3), 2),
            'used': round(swap.used / (1024**3), 2),
            'percent': round(swap.percent, 2)
        },
        'disk': {
            'total': round(disk.total / (1024**3), 2),
            'used': round(disk.used / (1024**3), 2),
            'free': round(disk.free / (1024**3), 2),
            'percent': round(disk.percent, 2),
            'read_bytes': disk_io.read_bytes if disk_io else 0,
            'write_bytes': disk_io.write_bytes if disk_io else 0
        },
        'network': {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        },
        'gpu': gpu_info,
        'system': {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'boot_time': boot_time,
            'uptime_seconds': round(uptime_seconds, 2),
            'uptime_formatted': format_uptime(uptime_seconds)
        },
        'top_processes': processes
    }

    return jsonify(data)


def get_gpu_info():
    """Tenta obter informações da GPU"""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_list = []
            for gpu in gpus:
                gpu_list.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': round(gpu.load * 100, 2),
                    'memory_used': round(gpu.memoryUsed, 2),
                    'memory_total': round(gpu.memoryTotal, 2),
                    'memory_percent': round((gpu.memoryUsed / gpu.memoryTotal) * 100, 2),
                    'temperature': gpu.temperature
                })
            return gpu_list
    except ImportError:
        pass
    except Exception as e:
        print(f"Erro ao obter info da GPU: {e}")

    return None


def format_uptime(seconds):
    """Formata tempo de uptime"""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)

    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


# ==================== GERENCIAMENTO DE EXPERIMENTOS ====================

@app.route('/api/experiments', methods=['POST'])
def create_experiment():
    """Recebe e salva um novo experimento, então executa o main.py como daemon"""
    try:
        # Recebe o JSON do frontend
        experiment_data = request.get_json()

        if not experiment_data:
            return jsonify({'error': 'Nenhum dado recebido'}), 400

        # Gera ID único baseado no timestamp
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        experiment_data['id'] = experiment_id
        experiment_data['received_at'] = datetime.now().isoformat()

        # Salva o JSON em arquivo
        filename = os.path.join(EXPERIMENTS_DIR, f"{experiment_id}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(experiment_data, f, indent=2, ensure_ascii=False)

        # Adiciona à lista em memória
        experiments_list.append(experiment_data)

        print(f"✓ Experimento recebido: {experiment_data.get('experiment_name', 'N/A')}")
        print(f"  ID: {experiment_id}")
        print(f"  Arquivo: {filename}")

        # Constrói o comando do main.py
        cmd = build_main_command(experiment_data)

        # MUDANÇA: Executa de forma síncrona para verificar se o daemon foi criado
        print(f"\n{'='*80}")
        print(f"🚀 Iniciando daemon para experimento {experiment_id}...")
        print(f"{'='*80}\n")

        # Registra o processo como em execução
        running_processes[experiment_id] = {
            'status': 'starting',
            'start_time': datetime.now().isoformat(),
            'command': cmd
        }

        # Executa o comando como daemon (bloqueante apenas para criar o processo)
        success, result = run_cmd_daemon(cmd, experiment_id)

        if not success:
            # ERRO ao criar o daemon
            running_processes[experiment_id]['status'] = 'failed'
            running_processes[experiment_id]['error'] = result
            running_processes[experiment_id]['end_time'] = datetime.now().isoformat()

            print(f"\n{'='*80}")
            print(f"❌ FALHA ao criar daemon para experimento {experiment_id}!")
            print(f"{'='*80}")
            print(f"Erro: {result}")
            print(f"{'='*80}\n")

            return jsonify({
                'status': 'error',
                'message': 'Falha ao criar processo daemon',
                'error': result,
                'id': experiment_id,
                'experiment_name': experiment_data.get('experiment_name', 'N/A'),
                'timestamp': experiment_data['received_at'],
                'command': cmd,
                'execution_status': 'failed'
            }), 500

        # SUCESSO ao criar o daemon
        # Atualiza informações do processo daemon
        running_processes[experiment_id].update({
            'status': 'running',
            'pid': result['pid'],
            'stdout_log': result['stdout_log'],
            'stderr_log': result['stderr_log']
        })

        print(f"\n{'='*80}")
        print(f"✅ Daemon criado COM SUCESSO!")
        print(f"{'='*80}")
        print(f"🆔 Experimento: {experiment_id}")
        print(f"🔢 PID: {result['pid']}")
        print(f"📝 Logs: {result['stdout_log']}")
        print(f"{'='*80}\n")

        # Inicia thread de monitoramento (não bloqueia)
        monitor_thread = threading.Thread(
            target=monitor_process_daemon,
            args=(experiment_id, result),
            daemon=True,
            name=f"Monitor-{experiment_id}"
        )
        monitor_thread.start()

        print(f"🔍 Thread de monitoramento iniciada: {monitor_thread.name}\n")

        # RESPOSTA DE SUCESSO com informações completas
        return jsonify({
            'status': 'success',
            'message': 'Experimento salvo e daemon iniciado com sucesso',
            'id': experiment_id,
            'experiment_name': experiment_data.get('experiment_name', 'N/A'),
            'timestamp': experiment_data['received_at'],
            'command': cmd,
            'execution_status': 'running',
            'mode': 'daemon',
            'daemon_info': {
                'pid': result['pid'],
                'stdout_log': result['stdout_log'],
                'stderr_log': result['stderr_log'],
                'created_at': datetime.now().isoformat()
            }
        }), 201

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()

        print(f"\n{'='*80}")
        print(f"❌ EXCEÇÃO ao processar experimento!")
        print(f"{'='*80}")
        print(f"Erro: {str(e)}")
        print(f"\nStack trace:")
        print(error_details)
        print(f"{'='*80}\n")

        return jsonify({
            'status': 'error',
            'message': f'Erro ao processar experimento: {str(e)}',
            'error_details': error_details
        }), 500

# Adicione esta rota após a rota DELETE

@app.route('/api/experiments/<experiment_id>/cancel', methods=['POST'])
def cancel_experiment(experiment_id):
    """Cancela/para a execução de um experimento em andamento"""
    try:
        # Verifica se o experimento existe
        if experiment_id not in running_processes:
            return jsonify({
                'status': 'error',
                'message': 'Experimento não encontrado ou não está em execução'
            }), 404

        proc_info = running_processes[experiment_id]
        current_status = proc_info.get('status')

        # Verifica se o processo ainda está rodando
        if current_status not in ['starting', 'running']:
            return jsonify({
                'status': 'error',
                'message': f'Experimento já finalizou com status: {current_status}',
                'current_status': current_status
            }), 400

        pid = proc_info.get('pid')

        if not pid:
            return jsonify({
                'status': 'error',
                'message': 'PID do processo não encontrado'
            }), 404

        # Tenta cancelar o processo
        try:
            # Primeiro tenta SIGTERM (encerramento gracioso)
            print(f"\n{'='*80}")
            print(f"🛑 CANCELANDO EXPERIMENTO")
            print(f"{'='*80}")
            print(f"🆔 Experimento: {experiment_id}")
            print(f"🔢 PID: {pid}")
            print(f"📊 Status atual: {current_status}")
            print(f"⚡ Enviando SIGTERM...")

            os.kill(pid, signal.SIGTERM)

            # Aguarda até 5 segundos para o processo terminar graciosamente
            import time
            for i in range(5):
                try:
                    os.kill(pid, 0)  # Verifica se ainda existe
                    time.sleep(1)
                except OSError:
                    # Processo terminou
                    break

            # Se ainda estiver rodando, força com SIGKILL
            try:
                os.kill(pid, 0)
                print(f"⚠️  Processo ainda ativo, forçando com SIGKILL...")
                os.kill(pid, signal.SIGKILL)
                time.sleep(1)
            except OSError:
                pass  # Processo já terminou

            # Atualiza o status
            running_processes[experiment_id]['status'] = 'cancelled'
            running_processes[experiment_id]['cancelled_at'] = datetime.now().isoformat()
            running_processes[experiment_id]['end_time'] = datetime.now().isoformat()

            print(f"✅ Experimento cancelado com sucesso!")
            print(f"{'='*80}\n")

            return jsonify({
                'status': 'success',
                'message': 'Experimento cancelado com sucesso',
                'experiment_id': experiment_id,
                'pid': pid,
                'cancelled_at': running_processes[experiment_id]['cancelled_at'],
                'previous_status': current_status,
                'new_status': 'cancelled'
            }), 200

        except OSError as e:
            if e.errno == 3:  # No such process
                # Processo já não existe
                running_processes[experiment_id]['status'] = 'completed'
                running_processes[experiment_id]['end_time'] = datetime.now().isoformat()

                print(f"ℹ️  Processo {pid} já não existe")
                print(f"{'='*80}\n")

                return jsonify({
                    'status': 'info',
                    'message': 'Processo já havia terminado',
                    'experiment_id': experiment_id,
                    'pid': pid,
                    'new_status': 'completed'
                }), 200
            else:
                raise

    except PermissionError:
        print(f"❌ Permissão negada para cancelar processo {pid}")
        print(f"{'='*80}\n")

        return jsonify({
            'status': 'error',
            'message': 'Permissão negada para cancelar o processo',
            'experiment_id': experiment_id,
            'pid': pid
        }), 403

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()

        print(f"\n{'='*80}")
        print(f"❌ ERRO ao cancelar experimento {experiment_id}!")
        print(f"{'='*80}")
        print(f"Erro: {str(e)}")
        print(f"\nStack trace:")
        print(error_details)
        print(f"{'='*80}\n")

        return jsonify({
            'status': 'error',
            'message': f'Erro ao cancelar experimento: {str(e)}',
            'error_details': error_details
        }), 500


@app.route('/api/experiments', methods=['GET'])
def list_experiments():
    """Lista todos os experimentos salvos"""
    try:
        experiments = []

        # Lê todos os arquivos JSON do diretório
        for filename in os.listdir(EXPERIMENTS_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(EXPERIMENTS_DIR, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    exp_data = json.load(f)
                    exp_id = exp_data.get('id')

                    # Verifica status atualizado do processo
                    execution_status = check_process_status(exp_id)

                    # Informações do processo
                    process_info = {}
                    if exp_id in running_processes:
                        proc = running_processes[exp_id]
                        process_info = {
                            'status': execution_status,
                            'pid': proc.get('pid'),
                            'start_time': proc.get('start_time'),
                            'end_time': proc.get('end_time')
                        }

                    # Retorna apenas informações resumidas
                    experiments.append({
                        'id': exp_id,
                        'experiment_name': exp_data.get('experiment_name'),
                        'timestamp': exp_data.get('timestamp'),
                        'received_at': exp_data.get('received_at'),
                        'model_type': exp_data.get('configuration', {}).get('model_type'),
                        'execution_status': execution_status,
                        'process': process_info if process_info else None
                    })

        # Ordena por timestamp (mais recentes primeiro)
        experiments.sort(key=lambda x: x.get('received_at', ''), reverse=True)

        return jsonify({
            'status': 'success',
            'count': len(experiments),
            'experiments': experiments
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erro ao listar experimentos: {str(e)}'
        }), 500


@app.route('/api/experiments/<experiment_id>', methods=['GET'])
def get_experiment(experiment_id):
    """Retorna um experimento específico"""
    try:
        filename = os.path.join(EXPERIMENTS_DIR, f"{experiment_id}.json")

        if not os.path.exists(filename):
            return jsonify({
                'status': 'error',
                'message': 'Experimento não encontrado'
            }), 404

        with open(filename, 'r', encoding='utf-8') as f:
            experiment_data = json.load(f)

        # Verifica status atualizado
        execution_status = check_process_status(experiment_id)

        # Adiciona informações de execução se disponível
        if experiment_id in running_processes:
            experiment_data['execution'] = running_processes[experiment_id].copy()
            experiment_data['execution']['current_status'] = execution_status

        return jsonify({
            'status': 'success',
            'experiment': experiment_data
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erro ao buscar experimento: {str(e)}'
        }), 500


@app.route('/api/experiments/<experiment_id>/status', methods=['GET'])
def get_experiment_status(experiment_id):
    """Retorna o status de execução de um experimento"""
    try:
        # Verifica status atualizado
        current_status = check_process_status(experiment_id)

        if experiment_id not in running_processes:
            return jsonify({
                'status': 'success',
                'execution_status': 'not_started',
                'message': 'Experimento não foi executado ainda'
            }), 200

        execution_data = running_processes[experiment_id].copy()
        execution_data['current_status'] = current_status

        return jsonify({
            'status': 'success',
            'execution': execution_data
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erro ao buscar status: {str(e)}'
        }), 500


# @app.route('/api/experiments/<experiment_id>/logs', methods=['GET'])
# def get_experiment_logs(experiment_id):
#     """Retorna os logs de um experimento"""
#     try:
#         log_type = request.args.get('type', 'stdout')  # stdout ou stderr
#         tail = request.args.get('tail', type=int)  # Últimas N linhas
#
#         if experiment_id not in running_processes:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'Logs não encontrados para este experimento'
#             }), 404
#
#         proc_info = running_processes[experiment_id]
#         log_file = proc_info.get(f'{log_type}_log')
#
#         if not log_file or not os.path.exists(log_file):
#             return jsonify({
#                 'status': 'error',
#                 'message': f'Arquivo de log {log_type} não encontrado'
#             }), 404
#
#         with open(log_file, 'r') as f:
#             if tail:
#                 lines = f.readlines()
#                 content = ''.join(lines[-tail:])
#             else:
#                 content = f.read()
#
#         return jsonify({
#             'status': 'success',
#             'experiment_id': experiment_id,
#             'log_type': log_type,
#             'log_file': log_file,
#             'content': content
#         }), 200
#
#     except Exception as e:
#         return jsonify({
#             'status': 'error',
#             'message': f'Erro ao buscar logs: {str(e)}'
#         }), 500


@app.route('/api/experiments/<experiment_id>', methods=['DELETE'])
def delete_experiment(experiment_id):
    """Deleta um experimento específico"""
    try:
        filename = os.path.join(EXPERIMENTS_DIR, f"{experiment_id}.json")

        if not os.path.exists(filename):
            return jsonify({
                'status': 'error',
                'message': 'Experimento não encontrado'
            }), 404

        # Se o processo ainda está rodando, tenta terminar
        if experiment_id in running_processes:
            proc_info = running_processes[experiment_id]
            pid = proc_info.get('pid')

            if pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                    print(f"Processo {pid} do experimento {experiment_id} terminado")
                except OSError:
                    pass  # Processo já terminou

        os.remove(filename)

        # Remove das estruturas em memória
        if experiment_id in running_processes:
            del running_processes[experiment_id]

        return jsonify({
            'status': 'success',
            'message': 'Experimento deletado com sucesso',
            'id': experiment_id
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erro ao deletar experimento: {str(e)}'
        }), 500


# ==================== RESULTADOS DOS EXPERIMENTOS ====================

@app.route('/api/experiments/results', methods=['GET'])
def get_all_experiments_results():
    """
    Retorna os resultados de todos os experimentos que possuem Results.json
    """
    try:
        results_list = []

        # Verifica se o diretório de outputs existe
        if not os.path.exists(OUTPUTS_DIR):
            return jsonify({
                'status': 'success',
                'message': 'Nenhum diretório de outputs encontrado',
                'count': 0,
                'results': []
            }), 200

        # Itera sobre todos os diretórios de experimentos
        for exp_dir in os.listdir(OUTPUTS_DIR):
            exp_path = os.path.join(OUTPUTS_DIR, exp_dir)

            # Verifica se é um diretório
            if not os.path.isdir(exp_path):
                continue

            # Extrai o ID do experimento (remove o prefixo 'exp_')
            if exp_dir.startswith('exp_'):
                experiment_id = exp_dir[4:]  # Remove 'exp_'
            else:
                experiment_id = exp_dir

            # Busca os resultados do experimento
            results = get_experiment_results(experiment_id)
            monitors = get_experiment_monitors(experiment_id)

            # Busca informações adicionais do experimento se existir no JSON
            experiment_info = None
            exp_json_path = os.path.join(EXPERIMENTS_DIR, f"{experiment_id}.json")
            if os.path.exists(exp_json_path):
                try:
                    with open(exp_json_path, 'r', encoding='utf-8') as f:
                        experiment_info = json.load(f)
                except:
                    pass

            # Adiciona à lista se houver resultados
            if results is not None:
                result_entry = {
                    'experiment_id': experiment_id,
                    'experiment_name': experiment_info.get('experiment_name') if experiment_info else None,
                    'timestamp': experiment_info.get('timestamp') if experiment_info else None,
                    'results': results,
                    'has_monitors': len(monitors) > 0,
                    'monitors_count': len(monitors)
                }
                results_list.append(result_entry)

        # Ordena por timestamp (mais recentes primeiro)
        results_list.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        return jsonify({
            'status': 'success',
            'count': len(results_list),
            'results': results_list
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erro ao buscar resultados: {str(e)}'
        }), 500


@app.route('/api/experiments/<experiment_id>/results', methods=['GET'])
def get_single_experiment_results(experiment_id):
    """
    Retorna os resultados detalhados de um experimento específico
    Inclui Results.json e todos os monitors
    """
    try:
        # Busca os resultados
        results = get_experiment_results(experiment_id)

        if results is None:
            return jsonify({
                'status': 'error',
                'message': 'Resultados não encontrados para este experimento'
            }), 404

        # Busca os monitors
        monitors = get_experiment_monitors(experiment_id)

        # Busca informações adicionais do experimento
        experiment_info = None
        exp_json_path = os.path.join(EXPERIMENTS_DIR, f"{experiment_id}.json")
        if os.path.exists(exp_json_path):
            try:
                with open(exp_json_path, 'r', encoding='utf-8') as f:
                    experiment_info = json.load(f)
            except:
                pass

        # Busca status de execução atualizado
        execution_status = check_process_status(experiment_id)
        execution_info = None
        if experiment_id in running_processes:
            execution_info = running_processes[experiment_id].copy()
            execution_info['current_status'] = execution_status

        response_data = {
            'status': 'success',
            'experiment_id': experiment_id,
            'experiment_info': experiment_info,
            'execution_status': execution_info,
            'results': results,
            'monitors': monitors
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erro ao buscar resultados do experimento: {str(e)}'
        }), 500


@app.route('/api/experiments/<experiment_id>/logs', methods=['GET'])
def get_experiment_logs(experiment_id):
    """Retorna os logs de um experimento"""
    try:
        log_type = request.args.get('type', 'stdout')  # stdout ou stderr
        tail = request.args.get('tail', type=int)  # Últimas N linhas

        if experiment_id not in running_processes:
            return jsonify({
                'status': 'error',
                'message': 'Logs não encontrados para este experimento'
            }), 404

        proc_info = running_processes[experiment_id]
        log_file = proc_info.get(f'{log_type}_log')

        if not log_file or not os.path.exists(log_file):
            return jsonify({
                'status': 'error',
                'message': f'Arquivo de log {log_type} não encontrado'
            }), 404

        with open(log_file, 'r') as f:
            if tail:
                lines = f.readlines()
                content = ''.join(lines[-tail:])
            else:
                content = f.read()

        return jsonify({
            'status': 'success',
            'experiment_id': experiment_id,
            'log_type': log_type,
            'log_file': log_file,
            'content': content
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erro ao buscar logs: {str(e)}'
        }), 500


@app.route('/api/experiments/<experiment_id>/training-progress', methods=['GET'])
def get_training_progress(experiment_id):
    """
    Retorna o progresso do treinamento em tempo real
    Busca os arquivos monitor_model_*_fold.json na pasta Monitor
    """
    try:
        monitor_dir = os.path.join(OUTPUTS_DIR, f"exp_{experiment_id}", "Monitor")

        if not os.path.exists(monitor_dir):
            return jsonify({
                'status': 'no_data',
                'message': 'Nenhum dado de monitoramento disponível ainda',
                'experiment_id': experiment_id
            }), 200

        # Busca todos os arquivos de monitor
        monitor_files = sorted([f for f in os.listdir(monitor_dir) if f.endswith('.json')])

        if not monitor_files:
            return jsonify({
                'status': 'no_data',
                'message': 'Nenhum arquivo de monitoramento encontrado',
                'experiment_id': experiment_id
            }), 200

        # Lê todos os monitors
        monitors_data = []
        total_epochs_completed = 0
        latest_metrics = None
        overall_progress = 0

        for monitor_file in monitor_files:
            monitor_path = os.path.join(monitor_dir, monitor_file)
            try:
                with open(monitor_path, 'r', encoding='utf-8') as f:
                    monitor_data = json.load(f)

                    # Extrai informações do nome do arquivo (ex: monitor_model_0_fold.json)
                    parts = monitor_file.replace('monitor_model_', '').replace('_fold.json', '')
                    fold_number = int(parts) if parts.isdigit() else None

                    monitors_data.append({
                        'fold': fold_number,
                        'filename': monitor_file,
                        'data': monitor_data
                    })

                    # Conta épocas completadas
                    if 'epochs' in monitor_data:
                        total_epochs_completed += len(monitor_data['epochs'])

                        # Pega as métricas mais recentes
                        if monitor_data['epochs']:
                            latest_metrics = monitor_data['epochs'][-1]['metrics']

            except Exception as e:
                print(f"Erro ao ler {monitor_file}: {str(e)}")
                continue

        # Calcula progresso geral
        if monitors_data:
            # Assume que todos os folds têm o mesmo número de épocas esperadas
            first_monitor = monitors_data[0]['data']
            if 'epochs' in first_monitor and first_monitor['epochs']:
                epochs_per_fold = len(first_monitor['epochs'])
                total_folds = len(monitors_data)
                expected_total_epochs = epochs_per_fold * total_folds
                overall_progress = (total_epochs_completed / expected_total_epochs * 100) if expected_total_epochs > 0 else 0

        return jsonify({
            'status': 'success',
            'experiment_id': experiment_id,
            'monitors': monitors_data,
            'summary': {
                'total_folds': len(monitors_data),
                'total_epochs_completed': total_epochs_completed,
                'overall_progress': round(overall_progress, 2),
                'latest_metrics': latest_metrics
            }
        }), 200

    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': f'Erro ao buscar progresso: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500



# ==================== INICIALIZAÇÃO ====================

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 SERVIDOR UNIFICADO - Monitoramento & Experimentos (DAEMON MODE)")
    print("=" * 70)
    print("\n📊 MONITORAMENTO DE SISTEMA:")
    print(f"   Sistema: {platform.system()} {platform.release()}")
    print(f"   CPU: {psutil.cpu_count(logical=False)} cores físicos, {psutil.cpu_count(logical=True)} lógicos")
    print(f"   Memória: {round(psutil.virtual_memory().total / (1024**3), 2)} GB")
    print(f"   Dashboard: http://localhost:5000/")
    print(f"   API: http://localhost:5000/api/system-info")

    print("\n🧪 GERENCIAMENTO DE EXPERIMENTOS:")
    print(f"   Diretório: {os.path.abspath(EXPERIMENTS_DIR)}")
    print(f"   Outputs: {os.path.abspath(OUTPUTS_DIR)}")
    print(f"   API: http://localhost:5000/api/experiments")
    print(f"   ⚡ Executa main.py como DAEMON em background")
    print(f"   📝 Logs salvos em: outputs/exp_<id>/logs/")

    print("\n📈 RESULTADOS:")
    print(f"   API Todos: http://localhost:5000/api/experiments/results")
    print(f"   API Individual: http://localhost:5000/api/experiments/<id>/results")
    print(f"   API Logs: http://localhost:5000/api/experiments/<id>/logs?type=stdout")

    print("\n" + "=" * 70)
    print("🌐 Servidor rodando em: http://localhost:5000")
    print("⚡ Pressione Ctrl+C para parar")
    print("=" * 70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
