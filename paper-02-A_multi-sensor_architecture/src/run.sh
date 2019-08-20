echo 'Activating env'
source ./.venv/bin/activate
echo "Env is:"
which python
echo 'Appended experiment'
python run_experiment_appended.py
echo 'Union experiment'
python run_experiment_union.py
