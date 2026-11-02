# Marine Fish Classifier: End-to-End MLOps on AWS (Exp 8)

A scikit-learn model predicts a fish's species from five body measurements:
**Atlantic Herring, Atlantic Mackerel, Atlantic Salmon, Bluefin Tuna, Clownfish, European Sardine**.

It goes through the complete MLOps lifecycle. ETL feeds a feature store, MLflow tracks the experiments, FastAPI serves the model, Docker packages it, and GitHub Actions runs CI/CD. Terraform creates the AWS infrastructure (EC2 + ECR), and Prometheus + Grafana monitor the running app.

```
Laptop                         GitHub Actions                               AWS ap-south-1 (built by Terraform)
ETL → train 3 configs ─git push─► tests ─► build + smoke test ─► deploy ─► ECR ─pull─► EC2 :80 (FastAPI container)
      MLflow picks best                       │                                           │ /metrics (every 5 s)
      models/fish_classifier.joblib           └─► GHCR (public, amd64+arm64)   Prometheus :9090 ─► Grafana :3000
```

## What changed vs. the MNIST reference project

| Part | MNIST (reference) | This project |
|---|---|---|
| Data | MNIST images | Synthetic fisheries survey CSV (1,818 rows, deliberately messy) |
| ETL | normalise pixels | clean names, drop missing/duplicates/impossible values, engineer 2 features |
| Model | Keras CNN (TensorFlow) | scikit-learn Pipeline (StandardScaler + classifier) |
| Configs compared | 2 | 3: depth-2 decision tree (weak baseline), logistic regression, random forest |
| Model file | `models/digit_cnn.keras` | `models/fish_classifier.joblib` |
| Input | PNG upload (multipart) | JSON body with 5 numbers |
| Frontend | drawing canvas | form with species presets and probability bars |
| Metric label | `digit` | `species` |
| Errors | 400 bad image, 413, 415, 422, 503 | 400 out-of-range value, 413 body > 10 KB, 415 not JSON, 422 missing/wrong type, 503 no model |

Everything else (Docker, CI/CD, Terraform, monitoring, scripts) follows the same architecture.

**Engineered features** (shared by training and serving in `app/preprocessing.py`):
- `condition_factor` = 100 × weight(g) / length(cm)³ (Fulton's K, a real fisheries measure)
- `body_ratio` = length / height (body shape)

## Folder structure

```
marine-fish-classifier/
├── src/data_preprocessing.py      # Exp 4: ETL -> data/processed/fish.npz (feature store)
├── src/train.py                   # Exp 4: 3 configs -> 3 MLflow runs -> best copied to models/
├── models/fish_classifier.joblib  # created by training; commit it (≈0.6 MB)
├── models/model_card.json         # metrics + versions of the deployed model
├── app/preprocessing.py           # SHARED feature code (prevents training-serving skew)
├── app/model.py                   # FishPredictor: loads the bundle, checks feature contract
├── app/main.py                    # FastAPI: /, /health, /predict, /metrics
├── app/static/index.html          # frontend
├── tests/                         # pytest (fake model + real-model quality gate)
├── scripts/                       # configure, samples, load test, AWS + monitoring deploy
├── infra/                         # Exp 6: Terraform (ECR, SG, IAM, EC2)
├── monitoring/                    # Exp 7: Prometheus + Grafana (provisioned dashboard)
├── .github/workflows/ci.yml       # Exp 2/3: test -> docker -> deploy
├── Dockerfile  .dockerignore  .gitignore  pytest.ini
├── requirements.txt               # training + dev (includes mlflow, pytest)
└── requirements-serve.txt         # serving only (what goes in the Docker image)
```

---

# Step-by-step guide

Commands are for **zsh on macOS**. Copy them one block at a time. They contain no inline `#` comments, so zsh won't misread them.

## Phase 0: Set up the folder (one time)

1. Unzip `marine-fish-classifier.zip` into `~/Desktop/MLOps/`.
2. Open Terminal and run:

```bash
cd ~/Desktop/MLOps/marine-fish-classifier
ls -a
```

You should see `.github`, `.gitignore` and `.dockerignore` (hidden files). If `.github` is missing, the unzip skipped hidden files, so unzip again from Terminal with `unzip`.

3. Put your GitHub username into the files that need it. It gets lowercased automatically, because image names must be lowercase:

```bash
bash scripts/configure.sh YourGitHubUsername
```

4. Make the parent `MLOps` repo ignore this project (it gets its own repo):

```bash
echo "marine-fish-classifier/" >> ../.gitignore
```

## Phase 1: Model development (Exp 4) — Sprint 1

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m src.data_preprocessing
python -m src.train
```

Expected output (numbers may differ slightly):

```
[transform] 1741 clean rows (dropped: 5 unknown species, 36 missing, 18 duplicates, 18 out of range)
  baseline_tree_depth2     val_acc≈0.50
  logistic_regression      val_acc≈0.94
  random_forest_200        val_acc≈0.96
Selected: random_forest_200
```

Open MLflow and compare the runs:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Go to http://127.0.0.1:5000. Select all 3 runs, click **Compare**, and look at `curve_val_accuracy` (the learning curve). Also check the `selected_for_deployment` tag. Press Ctrl+C to stop MLflow.

📸 **Screenshots:** terminal ETL + training output, MLflow runs list, compare view, the `selected_for_deployment=true` tag.

**Viva points**
- *ETL:* Extract (raw CSV) → Transform (clean + feature engineering) → Load (train/val/test split saved to `fish.npz`, our simple feature store).
- *Why 3 configs?* To compare them objectively. The depth-2 tree is a deliberately weak baseline (it can't separate 6 classes with only 4 leaves).
- *Why select on validation, not test?* The test set must stay untouched so it gives an unbiased final estimate.
- *What does MLflow log?* Params (hyperparameters, data sizes, library versions), metrics (accuracy, macro-F1, learning curve), and artifacts (model, confusion matrix, classification report).

## Phase 2: Prediction app — Sprint 2

```bash
python -m pytest -v
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

All tests should pass. Open http://127.0.0.1:8000, click a species preset, then press **Predict**. Try http://127.0.0.1:8000/docs too.

In a **second terminal tab**:

```bash
cd ~/Desktop/MLOps/marine-fish-classifier
source .venv/bin/activate
python scripts/export_samples.py --count 20
python scripts/predict_samples.py
```

Test the error handling:

```bash
curl -s -H "Content-Type: application/json" -d '{"length_cm":9,"weight_kg":-1,"height_cm":4,"fin_count":7,"water_temp_c":26}' http://127.0.0.1:8000/predict
curl -s -H "Content-Type: text/plain" -d "hello" http://127.0.0.1:8000/predict
```

The first command gives a 400 (`weight_kg` out of range) and the second a 415. Stop the server with Ctrl+C.

📸 **Screenshots:** pytest all green, web page with a prediction, `/docs`, `predict_samples.py` accuracy.

**Viva points**
- *Training-serving skew:* the API builds features with the **same function** (`app/preprocessing.py`) as the ETL. On top of that, `FishPredictor` refuses a model trained on a different feature list.
- *Fake model in tests:* `dependency_overrides` swaps in a fake predictor so the unit tests are fast and independent of the model. Two real-model tests act as a quality gate.
- *Why pin scikit-learn?* A pickled/joblib model should be loaded with the same library version that saved it.

## Phase 3: Docker (Exp 5) — Sprint 3

Open **Docker Desktop** first.

```bash
cd ~/Desktop/MLOps/marine-fish-classifier
docker build -t marine-fish-classifier:1.0 .
docker run -d --name marine-fish-classifier-api -p 8000:8000 marine-fish-classifier:1.0
docker ps
curl http://127.0.0.1:8000/health
```

Wait until `docker ps` shows **(healthy)**, then open http://127.0.0.1:8000.

Useful commands:

```bash
docker logs marine-fish-classifier-api
docker stop marine-fish-classifier-api
docker rm -f marine-fish-classifier-api
docker images
```

**Publish a multi-arch image to GHCR** (so the examiner's Windows/amd64 PC can run it):

1. GitHub → Settings → Developer settings → Personal access tokens → **Tokens (classic)** → Generate, with scopes `write:packages` and `repo`. Copy the token.
2. Run these (replace the token and the lowercase username):

```bash
export GHCR_TOKEN=paste_token_here
echo $GHCR_TOKEN | docker login ghcr.io -u manavnavadiya --password-stdin
docker buildx create --name multi --use
docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/manavnavadiya/marine-fish-classifier:1.0 -t ghcr.io/manavnavadiya/marine-fish-classifier:latest --push .
docker buildx imagetools inspect ghcr.io/manavnavadiya/marine-fish-classifier:1.0
```

If `docker buildx create` says the name `multi` already exists (from your MNIST project), run `docker buildx use multi` instead.

3. On GitHub, go to your profile → **Packages** → `marine-fish-classifier` → **Package settings**:
   - **Change visibility → Public**
   - **Manage Actions access → Add repository** → `marine-fish-classifier` → role **Write**. Without this, the Actions push fails with `permission_denied: write_package`.

The examiner runs one command (works on Windows, Mac and Linux):

```bash
docker run -p 8000:8000 ghcr.io/manavnavadiya/marine-fish-classifier:latest
```

Unlike TensorFlow, scikit-learn runs fine under emulation, so you can test the amd64 image on your Mac with `docker run --platform linux/amd64 ...`.

📸 **Screenshots:** `docker build` finished, `docker ps` showing healthy, app in browser from the container, GHCR package page (public, 2 platforms).

**Viva points**
- Dependencies are installed before the code is copied, so Docker caches that layer.
- The app runs as a non-root user (created before `COPY --chown`).
- The `HEALTHCHECK` checks `model_loaded`, not just HTTP 200.
- The serving image has no MLflow, pandas or tests (`requirements-serve.txt` + `.dockerignore`), so it stays small.
- A multi-arch image (amd64 + arm64) runs both on an Apple Silicon Mac and on Windows PCs and EC2.

## Phase 4: CI/CD with GitHub Actions (Exp 2, 3) — Sprint 4

1. Create an **empty** repo on GitHub named `marine-fish-classifier`, with no README or .gitignore.
2. Push the code:

```bash
cd ~/Desktop/MLOps/marine-fish-classifier
git init -b main
git add .
git status
git commit -m "Marine fish classifier: MLOps pipeline"
git remote add origin https://github.com/manavnavadiya/marine-fish-classifier.git
git push -u origin main
```

Before committing, make sure `git status` lists `models/fish_classifier.joblib` and `.github/workflows/ci.yml`, and does **not** list `.venv`, `mlruns` or `data`.

3. Open the repo → **Actions** tab. You should see **Unit tests** ✅ → **Build, smoke-test, publish** ✅. **Deploy to AWS** is skipped until Phase 5 is set up; that's expected.

4. **Show that a failing test blocks publishing** (for the report). Edit `tests/test_preprocessing.py`, change `pytest.approx(5.0)` to `pytest.approx(6.0)`, then:

```bash
git commit -am "Demo: failing test"
git push
```

Actions turns red at **Unit tests**, and the Docker job never runs. Change it back and push again:

```bash
git commit -am "Fix test"
git push
```

📸 **Screenshots:** green run with the jobs, the red run where `docker` was skipped, the GHCR package with the `sha-xxxxxxx` tag.

**Viva points**
- *CI* = every push is tested automatically. *CD* = after the tests pass, the image is built, smoke-tested as a real container, published and deployed.
- `needs:` creates the gates: test → docker → deploy.
- Pull requests get tested but never publish or deploy.
- The image is tagged with the commit SHA (traceable) and `latest`.

## Phase 5: AWS + Terraform (Exp 6) — Sprint 5

You already have the AWS CLI, Terraform, the `terraform-mlops` IAM user and the `mlops-key` key pair from your MNIST project, so this project reuses them. All resource names start with `marine-fish-classifier-`, so nothing clashes with the MNIST resources.

> 💰 **Cost tip:** if your MNIST EC2 is still running and you no longer need it, destroy it first: `cd ~/Desktop/MLOps/<mnist-folder>/infra` then `terraform destroy`.

```bash
cd ~/Desktop/MLOps/marine-fish-classifier
aws sts get-caller-identity
bash scripts/aws_deploy.sh
```

The script will:
1. write your IP into `terraform.tfvars`
2. reuse the key pair
3. run `terraform init/plan`
4. ask before applying (press Enter)
5. create the ECR repo, security group, IAM role + profile and EC2 (`t3.small`)
6. copy the image from GHCR to ECR
7. deploy over SSH
8. test the public IP

At the end, open the printed `http://<public-ip>`.

Then connect GitHub Actions to AWS:

```bash
gh auth status
bash scripts/setup_github_deploy.sh
```

If `gh auth status` says you're not logged in, run `gh auth login` first. The script reuses your `github-deployer` user and creates a new access key. AWS allows at most 2 keys per user; if you hit that limit, the script tells you how to delete an old one. It then saves the secrets and variables to the new repo.

Now trigger a full deploy:

```bash
git commit --allow-empty -m "Trigger deploy"
git push
```

In Actions, all three jobs should go green, with **Deploy to AWS** last.

📸 **Screenshots:** `terraform apply` output, EC2 + ECR in the AWS console (region Mumbai), the app on the public IP, green run with the 3 jobs.

**Viva points**
- *IaC:* infrastructure is declared in code, version-controlled, repeatable and destroyable (`plan` before `apply`).
- *Terraform creates 7 resources:* ECR, security group, IAM role, 2 policy attachments, instance profile and EC2.
- *Security:* port 80 is open to everyone. Ports 22, 3000 and 9090 are open only to your IP. EC2 pulls from ECR with an IAM role, so no keys are stored on the server.
- *Least privilege:* GitHub Actions uses a separate `github-deployer` user that can only push to ECR and send SSM commands.
- *SSM vs SSH:* CI deploys through AWS Systems Manager, so no SSH key is stored in GitHub.
- `lifecycle { ignore_changes = [ami] }` stops a newer AMI from replacing the server (and changing its IP).

## Phase 6: Monitoring (Exp 7) — Sprint 6

```bash
bash scripts/deploy_monitoring.sh
```

Open Grafana at `http://<public-ip>:3000` (login **admin / mlops-demo**). The "Marine Fish Classifier — Live" dashboard is the home page.

Generate traffic (about 15% of it deliberately bad):

```bash
source .venv/bin/activate
python scripts/load_test.py --url http://<public-ip> --requests 300
```

Watch the panels: requests/sec, `/predict` error %, p50/p95/p99 latency, predictions by species, and rejected requests by reason (`invalid_input`, `validation_error`, `unsupported_type`).

**Failure demo:**

```bash
ssh -i ~/.ssh/mlops-key.pem ec2-user@<public-ip>
docker stop marine-fish-classifier-api
```

Within about 10 s, **App up** turns red (DOWN). Bring it back:

```bash
docker start marine-fish-classifier-api
exit
```

You can also check Prometheus at `http://<public-ip>:9090/targets` (the target should show UP).

📸 **Screenshots:** Grafana during the load test, App up = DOWN after `docker stop`, Prometheus targets page.

**Viva points**
- *Prometheus pulls* (scrapes) `/metrics` every 5 s. *Grafana* queries Prometheus with PromQL and visualises it.
- *Counter vs Histogram vs Gauge:* request counts only go up; latency/confidence are distributions (used for p95); `model_loaded` goes up and down.
- *Label values are pre-created* so panels show 0 instead of "No data".
- *Endpoint label = route template,* so random URLs can't create unlimited time series (cardinality).
- *ML-specific metrics:* prediction distribution by species (a sudden skew can hint at data drift) and average confidence.

## Cleanup (when you're done and have your screenshots)

```bash
cd ~/Desktop/MLOps/marine-fish-classifier/infra
terraform destroy
```

Never delete `infra/terraform.tfstate` by hand: without it, Terraform can no longer manage (or destroy) the resources.

---

## Scrum framing (Exp 1)

**Product backlog:** ETL pipeline, model comparison, API, frontend, tests, container, registry publish, CI pipeline, IaC, cloud deploy, CD to cloud, monitoring, load/error testing, documentation.

**Sprints:** Phase 1 → Sprint 1 … Phase 6 → Sprint 6. Each sprint ends with a demoable increment (the deliverable column below).

| Sprint | Deliverable | Experiment |
|---|---|---|
| 1 | Trained, tracked, selected model | Exp 4 |
| 2 | Working web app + tests | — |
| 3 | Portable multi-arch image on GHCR | Exp 5 |
| 4 | Automated CI/CD pipeline | Exp 2, 3 |
| 5 | Live AWS deployment via Terraform | Exp 6 |
| 6 | Monitored ML app | Exp 7 |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `python3.11: command not found` | `brew install python@3.11` |
| `pip install` fails building numpy/scipy | Make sure the venv uses Python **3.11** (`python --version`), not 3.13 |
| `docker build`: `open Dockerfile: no such file` | `cd` into the project folder first |
| `COPY models/fish_classifier.joblib: not found` | Run Phase 1 training first |
| `repository name must be lowercase` | Run `bash scripts/configure.sh <username>`; use lowercase in commands |
| Actions: `denied: permission_denied: write_package` | GHCR package settings → Manage Actions access → add repo with **Write** → re-run failed jobs |
| Actions test job: real-model test fails | You changed data/model; retrain, commit `models/`, push |
| Deploy job skipped | Run `scripts/setup_github_deploy.sh` (it sets `EC2_INSTANCE_ID`) |
| `ERR_CONNECTION_REFUSED` on public IP | The container isn't running yet; wait for the deploy step |
| Grafana `:3000` times out | Run `bash scripts/deploy_monitoring.sh`; if your home IP changed, it re-opens the ports |
| `zsh: bad CPU type in executable: aws` | `brew install awscli` (arm64 build) |
| Model warning about scikit-learn version | Retrain with the pinned `requirements.txt` so the train and serve versions match |

When something fails, paste back: the command you ran, the full error output, and (for Actions) the failing step's log.
