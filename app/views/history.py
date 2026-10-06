import math

from flask import Blueprint, current_app, render_template, request

from app.models import Prediction

bp = Blueprint("history", __name__)


@bp.route("/history")
def index():
    page = max(request.args.get("page", 1, type=int), 1)
    size = current_app.config["HISTORY_PAGE_SIZE"]
    items, total = Prediction.page(page, size)
    return render_template("history.html", items=items, page=page, pages=max(math.ceil(total / size), 1),
                           total=total)
