import subprocess

def run_script(script_name):
    """Given a script name, run the script using subprocess."""
    try:
        print(f"Running {script_name}...")
        result = subprocess.run(['python', script_name], check=True)
        print(f"{script_name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")
        exit(1)

def main():

    # Veri Getirme (Fetch)
    run_script('data_fetch_epaticom.py')
    run_script('data_fetch_kitaplık.py')
    run_script('data_fetch_kb.py')

    # Veri Temizleme
    run_script('data_clean_epaticom.py')
    run_script('data_clean_kitaplık.py')
    run_script('data_clean_kitaplık_eng.py')

    # Veri Filtreleme
    run_script('data_filter_epaticom.py')
    run_script('data_filter_kitaplık.py')
    run_script('data_filter_kitaplık_eng.py')

    # Veri Hazırlama
    run_script('data_preparation.py')

    # Model Eğitimi
    run_script('model_training_basic.py')
    run_script('model_training_qa.py')
    run_script('model_training_text.py')

    # Flask Uygulamasını Çalıştırma
    print("Starting Flask application...")
    run_script('app.py')

if __name__ == '__main__':
    main()
